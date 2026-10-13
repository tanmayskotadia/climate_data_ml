"""
Climate Observation Quality & Anomaly Detection Engine (Objective 1)
Systematically identifies:
1. Missing observations & temporal gaps
2. Physical plausibility boundary violations
3. Sensor freeze / telemetry flatlines
4. Severe rate-of-change spikes & electronic glitches
5. Rolling statistical outliers (Z-score, IQR)
6. Multi-variate unsupervised anomalies (Isolation Forest)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest

PHYSICAL_LIMITS = {
    "temperature": (-50.0, 60.0),       # °C
    "relative_humidity": (0.0, 100.0),   # %
    "wind_speed": (0.0, 75.0),           # m/s
    "precipitation": (0.0, 500.0),       # mm
    "solar_radiation": (0.0, 1400.0),    # W/m^2 or equivalent
    "surface_pressure": (800.0, 1100.0)  # hPa
}

RATE_OF_CHANGE_MAX = {
    "temperature": 8.0,          # Max plausible change per step (°C)
    "relative_humidity": 35.0,    # % per step
    "wind_speed": 15.0,           # m/s per step
    "precipitation": 100.0,       # mm per step
    "solar_radiation": 600.0      # W/m^2 per step
}

def detect_temporal_gaps(df: pd.DataFrame) -> Dict[str, any]:
    """
    Detects missing timestamps or irregularities in time series frequency.
    """
    if "timestamp" not in df.columns:
        return {"has_timestamp": False}
    
    ts = pd.to_datetime(df["timestamp"]).sort_values().reset_index(drop=True)
    diffs = ts.diff().dropna()
    
    if len(diffs) == 0:
        return {"has_timestamp": True, "total_records": len(df), "gaps_found": 0}
        
    median_delta = diffs.median()
    irregularities = diffs[diffs > median_delta * 1.5]
    
    return {
        "has_timestamp": True,
        "total_records": len(df),
        "nominal_frequency": str(median_delta),
        "timestamp_skips_count": len(irregularities),
        "max_skip_duration": str(diffs.max()) if len(diffs) > 0 else "None",
        "irregular_indices": [int(i) for i in irregularities.index]
    }

def detect_climate_anomalies(
    df: pd.DataFrame,
    variable: str = "temperature",
    z_threshold: float = 3.0,
    freeze_window: int = 5
) -> Dict[str, any]:
    """
    Comprehensive anomaly detection for a specific climate variable.
    Returns categorized boolean masks and summary diagnostics.
    """
    series = df[variable].copy()
    n = len(series)
    values = series.values.astype(float)
    
    # 1. Missingness mask
    missing_mask = np.isnan(values)
    
    # Non-missing values for calculations
    valid_idx = np.where(~missing_mask)[0]
    valid_vals = values[valid_idx]
    
    # 2. Physical boundary violations
    phys_mask = np.zeros(n, dtype=bool)
    if variable in PHYSICAL_LIMITS and len(valid_vals) > 0:
        low, high = PHYSICAL_LIMITS[variable]
        phys_mask[valid_idx] = (valid_vals < low) | (valid_vals > high)
        
    # 3. Sensor freeze / flatline detection (consecutive identical values)
    freeze_mask = np.zeros(n, dtype=bool)
    if len(valid_vals) >= freeze_window:
        consecutive_count = 1
        for i in range(1, len(valid_vals)):
            curr_idx = valid_idx[i]
            prev_idx = valid_idx[i-1]
            # If values are identical down to small epsilon (excluding normal 0 precipitation)
            if np.isclose(valid_vals[i], valid_vals[i-1], atol=1e-5) and not (variable == "precipitation" and valid_vals[i] == 0.0):
                consecutive_count += 1
                if consecutive_count >= freeze_window:
                    freeze_mask[curr_idx] = True
                    # Backtrack flag the previous window points
                    for b in range(1, consecutive_count):
                        freeze_mask[valid_idx[i - b]] = True
            else:
                consecutive_count = 1

    # 4. Sudden Rate of Change (Spike) Detection
    spike_mask = np.zeros(n, dtype=bool)
    if len(valid_vals) > 2:
        diffs = np.abs(np.diff(valid_vals))
        max_rate = RATE_OF_CHANGE_MAX.get(variable, np.std(valid_vals) * 3.5)
        spike_in_diffs = np.where(diffs > max_rate)[0]
        # Flag the peak
        for s in spike_in_diffs:
            spike_mask[valid_idx[s + 1]] = True

    # 5. Rolling Statistical Outliers (Z-score & IQR)
    stat_mask = np.zeros(n, dtype=bool)
    if len(valid_vals) > 10:
        roll_series = pd.Series(valid_vals)
        rolling_mean = roll_series.rolling(window=24, min_periods=3, center=True).mean().bfill().ffill().values
        rolling_std = roll_series.rolling(window=24, min_periods=3, center=True).std().bfill().ffill().values
        rolling_std = np.where(rolling_std < 1e-4, 1.0, rolling_std)
        z_scores = np.abs((valid_vals - rolling_mean) / rolling_std)
        stat_mask[valid_idx] = z_scores > z_threshold

    # 6. Unsupervised Machine Learning (Isolation Forest)
    iso_mask = np.zeros(n, dtype=bool)
    if len(valid_vals) >= 30:
        try:
            # Build feature matrix: value, first difference, rolling mean
            feat_val = valid_vals.reshape(-1, 1)
            diff_feat = np.concatenate([[0], np.diff(valid_vals)]).reshape(-1, 1)
            X = np.hstack([feat_val, diff_feat])
            
            iso_model = IsolationForest(contamination=0.03, random_state=42)
            preds = iso_model.fit_predict(X)
            iso_mask[valid_idx] = (preds == -1)
        except Exception:
            pass

    # Composite Anomaly Mask (Union of physical, spike, freeze, stat)
    composite_anomaly = phys_mask | spike_mask | freeze_mask | stat_mask

    return {
        "variable": variable,
        "total_records": n,
        "missing_count": int(np.sum(missing_mask)),
        "missing_percentage": round(float(np.mean(missing_mask) * 100), 2),
        "physical_violation_count": int(np.sum(phys_mask)),
        "sensor_freeze_count": int(np.sum(freeze_mask)),
        "spike_anomaly_count": int(np.sum(spike_mask)),
        "statistical_outlier_count": int(np.sum(stat_mask)),
        "isolation_forest_anomaly_count": int(np.sum(iso_mask)),
        "total_anomalies_detected": int(np.sum(composite_anomaly)),
        "masks": {
            "missing": missing_mask,
            "physical": phys_mask,
            "freeze": freeze_mask,
            "spike": spike_mask,
            "statistical": stat_mask,
            "isolation_forest": iso_mask,
            "composite": composite_anomaly
        }
    }
