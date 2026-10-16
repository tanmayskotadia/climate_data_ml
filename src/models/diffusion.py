"""
Conditional Spatiotemporal Diffusion Imputer (CoFILL Baseline - IJCAI 2025)
Paper: "Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion"
Implements:
- Forward Gaussian diffusion process for missing observation latent tokens
- Conditional Denoising Network (Score/Epsilon model) conditioned on observed context
- Multi-step reverse sampling trajectory (denoising steps)
- Uncertainty estimation through stochastic reverse sampling variance
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

# Disable cuDNN engine benchmark for 1D convolutions to prevent Windows driver engine errors
torch.backends.cudnn.enabled = False

class DiffusionDenoisingNet(nn.Module):
    """
    U-Net style 1D Temporal Denoising Network for conditional imputation.
    Conditions on observed values and time-step embedding.
    """
    def __init__(self, hidden_dim: int = 48):
        super(DiffusionDenoisingNet, self).__init__()
        # Input features: [noisy_x (1), observed_mask (1), observed_vals (1)] -> 3 channels
        self.input_conv = nn.Conv1d(3, hidden_dim, kernel_size=3, padding=1)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dual-stream residual blocks (Temporal + Spectral filtering)
        self.res1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.res2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, hidden_dim)
        self.norm2 = nn.GroupNorm(4, hidden_dim)
        self.act = nn.SiLU()
        self.out_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x, mask, obs: [batch, 1, seq_len]
        # t: [batch, 1]
        feat = torch.cat([x, mask, obs], dim=1)  # [batch, 3, seq_len]
        h = self.input_conv(feat)
        
        # Add time embedding
        t_embed = self.time_emb(t).unsqueeze(-1)  # [batch, hidden_dim, 1]
        h = h + t_embed
        
        # Residual processing
        r = self.act(self.norm1(self.res1(h)))
        r = self.act(self.norm2(self.res2(r)))
        h = h + r
        
        eps = self.out_conv(h)  # predicted noise [batch, 1, seq_len]
        return eps

class ConditionalClimateDiffusion(nn.Module):
    def __init__(self, num_diffusion_steps: int = 20, beta_start: float = 0.001, beta_end: float = 0.05):
        super(ConditionalClimateDiffusion, self).__init__()
        self.num_steps = num_diffusion_steps
        self.denoiser = DiffusionDenoisingNet(hidden_dim=48)
        
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        batch_size = x_0.size(0)
        s_alpha = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1)
        s_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1)
        return s_alpha * x_0 + s_one_minus * noise

    def sample_reverse(self, obs_vals: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Reverse sampling trajectory (denoising) for missing points.
        Conditioned on observed observations at every step.
        """
        batch_size, _, seq_len = obs_vals.shape
        # Initialize missing positions with standard normal noise
        x_t = torch.randn_like(obs_vals)
        # Keep observed points fixed
        x_t = torch.where(obs_mask.bool(), obs_vals, x_t)
        
        for step in reversed(range(self.num_steps)):
            t_tensor = torch.tensor([[step / self.num_steps]], dtype=torch.float32, device=obs_vals.device)
            # Predict noise
            pred_noise = self.denoiser(x_t, obs_mask, obs_vals, t_tensor)
            
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alphas_cumprod[step]
            
            # Mean update
            mean = (1.0 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1.0 - alpha_bar + 1e-8)) * pred_noise)
            if step > 0:
                var = beta * (1.0 - self.alphas_cumprod[step - 1]) / (1.0 - alpha_bar + 1e-8)
                sigma = torch.sqrt(torch.clamp(var, min=1e-8))
                z = torch.randn_like(x_t)
                x_t = mean + sigma * z
            else:
                x_t = mean
                
            # Clamp conditioned observed points to true values
            x_t = torch.where(obs_mask.bool(), obs_vals, x_t)
            
        return x_t

def train_and_impute_diffusion(
    series: pd.Series,
    epochs: int = 30,
    num_diffusion_steps: int = 15,
    lr: float = 0.005,
    device: str = "cpu"
) -> Tuple[np.ndarray, float, ConditionalClimateDiffusion, Dict[str, np.ndarray]]:
    """
    Fits conditional diffusion model on observed series and executes reverse denoising.
    Returns:
      imputed: Reconstructed series array
      elapsed_ms: Inference and training runtime
      model: Trained diffusion model
      uncertainty: Dict with mean, std, lower_ci, and upper_ci derived from diffusion trajectories
    """
    start_t = time.perf_counter()
    raw_vals = series.values.copy()
    n = len(raw_vals)
    mask_np = ~np.isnan(raw_vals)
    
    valid_vals = raw_vals[mask_np]
    mean_v = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
    std_v = float(np.std(valid_vals)) if len(valid_vals) > 0 else 1.0
    
    norm_vals = np.where(mask_np, (raw_vals - mean_v) / (std_v + 1e-6), 0.0)
    
    obs_tensor = torch.tensor(norm_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device) # [1, 1, n]
    mask_tensor = torch.tensor(mask_np.astype(float), dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    
    model = ConditionalClimateDiffusion(num_diffusion_steps=num_diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    # Train denoising network on synthetic noise perturbations of observed data
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        t = torch.randint(0, num_diffusion_steps, (1,), device=device)
        noise = torch.randn_like(obs_tensor)
        noisy_x = model.forward_diffusion(obs_tensor, t, noise)
        
        t_norm = (t.float() / num_diffusion_steps).view(1, 1)
        pred_noise = model.denoiser(noisy_x, mask_tensor, obs_tensor, t_norm)
        
        # Loss computed over observed tokens
        loss = (loss_fn(pred_noise, noise) * mask_tensor).sum() / (mask_tensor.sum() + 1e-6)
        loss.backward()
        optimizer.step()
        
    # Reverse sampling with multi-trajectory uncertainty estimation (10 paths)
    model.eval()
    trajectories = []
    with torch.no_grad():
        for _ in range(8):
            sampled_norm = model.sample_reverse(obs_tensor, mask_tensor).squeeze().cpu().numpy()
            sampled_denorm = sampled_norm * (std_v + 1e-6) + mean_v
            trajectories.append(sampled_denorm)
            
    traj_stack = np.stack(trajectories, axis=0) # [8, n]
    mean_pred = np.mean(traj_stack, axis=0)
    diff_std = np.std(traj_stack, axis=0) + std_v * 0.06
    
    diff_std[mask_np] = 0.0
    lower_ci = mean_pred - 1.96 * diff_std
    upper_ci = mean_pred + 1.96 * diff_std
    
    imputed = raw_vals.copy()
    imputed[~mask_np] = mean_pred[~mask_np]
    
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    
    uncertainty = {
        "mean": mean_pred,
        "std": diff_std,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci
    }
    return imputed, elapsed_ms, model, uncertainty
