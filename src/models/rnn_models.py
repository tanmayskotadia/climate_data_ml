"""
Deep Learning Recurrent Imputation Models (Objective 2 Baselines 2 & 3)
Implements:
- Bidirectional LSTM Imputer (ClimateBiLSTM)
- Bidirectional GRU Imputer (ClimateBiGRU)
Features:
- Mask-aware input representation [value, mask_indicator, time_feature]
- Bidirectional context propagation
- Monte Carlo Dropout support for uncertainty estimation
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

class ClimateBiLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(ClimateBiLSTM, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        h_proj = self.input_proj(x)
        out, _ = self.lstm(h_proj)
        out = self.dropout(out)
        pred = self.fc_out(out)
        return pred.squeeze(-1)

class ClimateBiGRU(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(ClimateBiGRU, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_proj = self.input_proj(x)
        out, _ = self.gru(h_proj)
        out = self.dropout(out)
        pred = self.fc_out(out)
        return pred.squeeze(-1)

def prepare_rnn_tensor(series: pd.Series, mean_val: float, std_val: float) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Constructs mask-aware input feature tensor for RNNs.
    Features:
      1. Normalized value (0.0 if missing)
      2. Binary mask (1 if observed, 0 if missing)
      3. Normalized time coordinate [0, 1]
    """
    raw_vals = series.values.copy()
    n = len(raw_vals)
    mask = ~np.isnan(raw_vals)  # 1 = observed, 0 = missing
    
    # Fill missing with 0 for model input
    clean_vals = np.where(mask, raw_vals, mean_val)
    norm_vals = (clean_vals - mean_val) / (std_val + 1e-6)
    norm_vals = np.where(mask, norm_vals, 0.0)
    
    time_coords = np.linspace(0, 1, n)
    
    feat = np.stack([norm_vals, mask.astype(float), time_coords], axis=-1)
    tensor_x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # [1, n, 3]
    return tensor_x, mask, raw_vals

def train_and_impute_rnn(
    model: nn.Module,
    series: pd.Series,
    epochs: int = 50,
    lr: float = 0.005,
    device: str = "cpu"
) -> Tuple[np.ndarray, float, nn.Module]:
    """
    Trains the RNN model on observed data points and performs imputation.
    Returns:
      imputed_values: Complete array with reconstructed values replacing missing points.
      elapsed_ms: Execution time in milliseconds.
      model: Trained model (for subsequent Monte Carlo Dropout uncertainty estimation).
    """
    start_t = time.perf_counter()
    vals = series.values
    valid_vals = vals[~np.isnan(vals)]
    mean_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
    std_val = float(np.std(valid_vals)) if len(valid_vals) > 0 else 1.0
    
    tensor_x, mask, raw_vals = prepare_rnn_tensor(series, mean_val, std_val)
    tensor_x = tensor_x.to(device)
    
    target = torch.tensor(np.where(mask, (raw_vals - mean_val) / (std_val + 1e-6), 0.0), dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    
    # Fast adaptation training on observed sequence
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(tensor_x)
        loss = (loss_fn(preds, target) * mask_t).sum() / (mask_t.sum() + 1e-6)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Inference
    model.eval()
    with torch.no_grad():
        norm_preds = model(tensor_x).squeeze(0).cpu().numpy()
        pred_vals = norm_preds * (std_val + 1e-6) + mean_val
    
    # Replace only missing values
    imputed = raw_vals.copy()
    imputed[~mask] = pred_vals[~mask]
    
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return imputed, elapsed_ms, model
