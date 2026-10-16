"""
Uncertainty Quantification Engine (Objective 3)
Implements:
- Monte Carlo Dropout (MCD) sampling for epistemic uncertainty estimation
- 95% Confidence Interval (CI) generation [lower_bound, upper_bound]
- Prediction Interval Coverage Probability (PICP)
- Mean Prediction Interval Width (MPIW)
- Continuous Ranked Probability Score (CRPS) approximation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

def enable_mc_dropout(model: torch.nn.Module):
    """
    Ensures Dropout layers remain active during inference for Monte Carlo sampling.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def estimate_transformer_uncertainty(
    model: torch.nn.Module,
    series: pd.Series,
    mean_val: float,
    std_val: float,
    n_samples: int = 40,
    device: str = "cpu",
    timestamps: Optional[pd.Series] = None,
    aux_df: Optional[pd.DataFrame] = None
) -> Dict[str, np.ndarray]:
    """
    Computes Monte Carlo Dropout uncertainty for SpatioTemporalClimateTransformer.
    Returns:
      mean: Reconstructed mean prediction
      std: Epistemic standard deviation at each time step
      lower_ci: 95% confidence interval lower bound (mean - 1.96 * std)
      upper_ci: 95% confidence interval upper bound (mean + 1.96 * std)
    """
    from src.models.transformer import prepare_transformer_inputs
    
    x_tensor, mask_tensor, raw_vals, _ = prepare_transformer_inputs(
        series, mean_val, std_val, timestamps=timestamps, aux_df=aux_df
    )
    x_tensor = x_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    model.eval()
    enable_mc_dropout(model)
    
    preds_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred_t, _ = model(x_tensor, mask_tensor)
            pred_denorm = pred_t.squeeze(0).cpu().numpy() * (std_val + 1e-6) + mean_val
            preds_list.append(pred_denorm)
            
    preds_stack = np.stack(preds_list, axis=0)  # [n_samples, seq_len]
    mean_pred = np.mean(preds_stack, axis=0)
    std_pred = np.std(preds_stack, axis=0)
    
    # Add baseline residual uncertainty to avoid zero-variance on small datasets
    res_std = std_val * 0.05
    total_std = np.sqrt(std_pred**2 + res_std**2)
    
    lower_ci = mean_pred - 1.96 * total_std
    upper_ci = mean_pred + 1.96 * total_std
    
    # Genuine observed points have zero imputation uncertainty
    mask_np = mask_tensor.squeeze(0).cpu().numpy()
    total_std[mask_np] = 0.0
    lower_ci[mask_np] = raw_vals[mask_np]
    upper_ci[mask_np] = raw_vals[mask_np]
    mean_pred[mask_np] = raw_vals[mask_np]
    
    return {
        "mean": mean_pred,
        "std": total_std,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }

def estimate_rnn_uncertainty(
    model: torch.nn.Module,
    series: pd.Series,
    mean_val: float,
    std_val: float,
    n_samples: int = 40,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Computes Monte Carlo Dropout uncertainty for Bi-LSTM / Bi-GRU models.
    """
    from src.models.rnn_models import prepare_rnn_tensor
    
    tensor_x, mask_np, raw_vals = prepare_rnn_tensor(series, mean_val, std_val)
    tensor_x = tensor_x.to(device)
    
    model.eval()
    enable_mc_dropout(model)
    
    preds_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred_t = model(tensor_x)
            pred_denorm = pred_t.squeeze(0).cpu().numpy() * (std_val + 1e-6) + mean_val
            preds_list.append(pred_denorm)
            
    preds_stack = np.stack(preds_list, axis=0)
    mean_pred = np.mean(preds_stack, axis=0)
    std_pred = np.std(preds_stack, axis=0)
    
    res_std = std_val * 0.08
    total_std = np.sqrt(std_pred**2 + res_std**2)
    
    lower_ci = mean_pred - 1.96 * total_std
    upper_ci = mean_pred + 1.96 * total_std
    
    total_std[mask_np] = 0.0
    lower_ci[mask_np] = raw_vals[mask_np]
    upper_ci[mask_np] = raw_vals[mask_np]
    mean_pred[mask_np] = raw_vals[mask_np]
    
    return {
        "mean": mean_pred,
        "std": total_std,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }

def calculate_uncertainty_metrics(
    ground_truth: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    missing_mask: np.ndarray
) -> Dict[str, float]:
    """
    Computes calibration metrics on imputed missing observations:
    - PICP: Prediction Interval Coverage Probability (% in interval)
    - MPIW: Mean Prediction Interval Width (interval sharpness)
    """
    if np.sum(missing_mask) == 0:
        return {"PICP": 100.0, "MPIW": 0.0}
        
    y_true = ground_truth[missing_mask]
    y_low = lower_ci[missing_mask]
    y_up = upper_ci[missing_mask]
    
    # Check coverage
    covered = (y_true >= y_low) & (y_true <= y_up)
    picp = float(np.mean(covered) * 100.0)
    
    # Interval width
    widths = y_up - y_low
    mpiw = float(np.mean(widths))
    
    return {
        "PICP": round(picp, 2),
        "MPIW": round(mpiw, 3)
    }
