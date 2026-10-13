"""
Evaluation Metrics Suite for Climate Observation Reconstruction
Computes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R-squared (R2 Score)
- MAPE (Mean Absolute Percentage Error)
- CRPS (Continuous Ranked Probability Score for normal distribution approx)
"""

import numpy as np
from typing import Dict

def compute_reconstruction_metrics(
    ground_truth: np.ndarray,
    reconstructed: np.ndarray,
    missing_mask: np.ndarray,
    uncertainty_std: np.ndarray = None,
    elapsed_ms: float = 0.0
) -> Dict[str, float]:
    """
    Evaluates reconstruction performance strictly on missing positions.
    """
    if np.sum(missing_mask) == 0:
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "R2": 1.0,
            "MAPE": 0.0,
            "CRPS": 0.0,
            "Latency_ms": round(elapsed_ms, 2)
        }
        
    y_true = ground_truth[missing_mask]
    y_pred = reconstructed[missing_mask]
    
    # Filter out any non-finite values if any
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(valid) == 0:
        return {"MAE": 999.0, "RMSE": 999.0, "R2": -1.0, "MAPE": 100.0, "CRPS": 999.0, "Latency_ms": round(elapsed_ms, 2)}
        
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    
    # R2 Score
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8))) if ss_tot > 1e-8 else 0.0
    
    # MAPE
    denom = np.where(np.abs(y_true) < 1e-2, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    
    # CRPS calculation under Gaussian predictive distribution assumption:
    # CRPS(N(mu, sigma^2), y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    crps = mae  # default to MAE if std is absent
    if uncertainty_std is not None:
        sigma = uncertainty_std[missing_mask][valid]
        sigma = np.clip(sigma, 1e-4, 50.0)
        z = (y_pred - y_true) / sigma
        # Approx standard normal CDF and PDF
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        # erf approx for Phi
        from scipy.special import erf
        Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        crps_arr = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
        crps = float(np.mean(np.abs(crps_arr)))
        
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "CRPS": round(crps, 3),
        "Latency_ms": round(elapsed_ms, 2)
    }
