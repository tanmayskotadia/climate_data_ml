"""
Traditional Statistical Interpolation Models (Objective 2 Baseline 1)
Implements:
- Linear Interpolation
- Cubic Spline Interpolation
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
- KNN Imputation
"""

import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline
from sklearn.impute import KNNImputer
from typing import Dict, Tuple

def run_linear_interpolation(series: pd.Series) -> Tuple[np.ndarray, float]:
    """
    Performs standard linear interpolation.
    """
    start_t = time.perf_counter()
    imputed = series.interpolate(method="linear").bfill().ffill().values
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return imputed, elapsed_ms

def run_cubic_spline_interpolation(series: pd.Series) -> Tuple[np.ndarray, float]:
    """
    Performs Cubic Spline interpolation with fallback for edge extrapolation.
    """
    start_t = time.perf_counter()
    vals = series.values.copy()
    valid_idx = np.where(~np.isnan(vals))[0]
    
    if len(valid_idx) < 4:
        # Fallback to linear if fewer than 4 points
        return run_linear_interpolation(series)
    
    cs = CubicSpline(valid_idx, vals[valid_idx], extrapolate=True)
    all_idx = np.arange(len(vals))
    imputed = cs(all_idx)
    
    # Clip extreme runaway spline oscillations to reasonable percentile bounds
    v_min, v_max = np.percentile(vals[valid_idx], [0.5, 99.5])
    span = v_max - v_min
    imputed = np.clip(imputed, v_min - span * 0.5, v_max + span * 0.5)
    
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return imputed, elapsed_ms

def run_pchip_interpolation(series: pd.Series) -> Tuple[np.ndarray, float]:
    """
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
    Preserves monotonicity and avoids the wild oscillations of raw splines.
    """
    start_t = time.perf_counter()
    vals = series.values.copy()
    valid_idx = np.where(~np.isnan(vals))[0]
    
    if len(valid_idx) < 3:
        return run_linear_interpolation(series)
    
    pchip = PchipInterpolator(valid_idx, vals[valid_idx], extrapolate=True)
    all_idx = np.arange(len(vals))
    imputed = pchip(all_idx)
    
    v_min, v_max = np.percentile(vals[valid_idx], [0.5, 99.5])
    span = v_max - v_min
    imputed = np.clip(imputed, v_min - span * 0.5, v_max + span * 0.5)
    
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return imputed, elapsed_ms

def run_knn_imputation(series: pd.Series, n_neighbors: int = 5) -> Tuple[np.ndarray, float]:
    """
    K-Nearest Neighbors Imputer using sliding window embedding.
    """
    start_t = time.perf_counter()
    vals = series.values.copy()
    n = len(vals)
    
    # Construct tabular lag feature representation for KNN
    window = 5
    features = []
    for i in range(n):
        row = []
        for offset in range(-window, window + 1):
            idx = i + offset
            if 0 <= idx < n:
                row.append(vals[idx])
            else:
                row.append(np.nan)
        features.append(row)
    
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    X_imputed = imputer.fit_transform(features)
    imputed = X_imputed[:, window]  # Central value
    
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return imputed, elapsed_ms
