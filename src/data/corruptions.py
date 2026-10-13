"""
Climate Data Corruption & Missingness Injection Module
Simulates real-world data quality issues encountered in climate archives:
- Random Missing at Random (MCAR)
- Structural Long Continuous Block Gaps (Power outage, communication breakdown)
- Sensor Freeze / Stuck Flatline (Sensor mechanical or ADC lockup)
- Extreme Sensor Spikes / Glitches (Electrical surge, voltage fluctuation)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

def inject_missingness_and_anomalies(
    df: pd.DataFrame,
    variable: str = "temperature",
    scenario: str = "random_20",
    seed: int = 42
) -> Tuple[pd.Series, np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Args:
        df: Input DataFrame with clean climate observations.
        variable: Climate variable column to corrupt.
        scenario: One of:
            - 'random_10': 10% MCAR missing values
            - 'random_20': 20% MCAR missing values
            - 'random_40': 40% MCAR missing values
            - 'block_24h': Continuous block gap of 24 consecutive hours
            - 'block_72h': Continuous block gap of 72 consecutive hours
            - 'sensor_freeze': Stuck sensor flatlining for 36 hours
            - 'sensor_spikes': Random severe electrical spikes/outliers
            - 'compound': Block gap + sensor freeze + intermittent spikes
        seed: Random seed for reproducibility.

    Returns:
        corrupted_series: Series containing np.nan for missing values or corrupted values.
        missing_mask: Boolean array where True indicates the value is MISSING (to be reconstructed).
        anomaly_mask: Boolean array where True indicates the value is an ANOMALOUS artifact.
        metadata: Dictionary with injection metadata (indices, gap lengths, etc.).
    """
    np.random.seed(seed)
    n = len(df)
    original_values = df[variable].values.copy()
    corrupted_values = original_values.copy().astype(float)
    
    missing_mask = np.zeros(n, dtype=bool)
    anomaly_mask = np.zeros(n, dtype=bool)
    meta = {"scenario": scenario, "variable": variable, "total_points": n}

    if scenario == "random_10":
        rate = 0.10
        idx = np.random.choice(n, size=int(n * rate), replace=False)
        missing_mask[idx] = True
        corrupted_values[idx] = np.nan
        meta["missing_count"] = len(idx)

    elif scenario == "random_20":
        rate = 0.20
        idx = np.random.choice(n, size=int(n * rate), replace=False)
        missing_mask[idx] = True
        corrupted_values[idx] = np.nan
        meta["missing_count"] = len(idx)

    elif scenario == "random_40":
        rate = 0.40
        idx = np.random.choice(n, size=int(n * rate), replace=False)
        missing_mask[idx] = True
        corrupted_values[idx] = np.nan
        meta["missing_count"] = len(idx)

    elif scenario == "block_24h":
        block_len = min(24, n // 2)
        start_idx = np.random.randint(n // 4, n * 3 // 4 - block_len)
        missing_mask[start_idx : start_idx + block_len] = True
        corrupted_values[start_idx : start_idx + block_len] = np.nan
        meta["block_start"] = int(start_idx)
        meta["block_len"] = int(block_len)

    elif scenario == "block_72h":
        block_len = min(72, n // 2)
        start_idx = np.random.randint(n // 4, n * 3 // 4 - block_len)
        missing_mask[start_idx : start_idx + block_len] = True
        corrupted_values[start_idx : start_idx + block_len] = np.nan
        meta["block_start"] = int(start_idx)
        meta["block_len"] = int(block_len)

    elif scenario == "sensor_freeze":
        freeze_len = min(36, n // 3)
        start_idx = np.random.randint(n // 4, n * 3 // 4 - freeze_len)
        frozen_val = original_values[start_idx]
        corrupted_values[start_idx : start_idx + freeze_len] = frozen_val
        anomaly_mask[start_idx + 1 : start_idx + freeze_len] = True
        meta["freeze_start"] = int(start_idx)
        meta["freeze_len"] = int(freeze_len)
        meta["frozen_val"] = float(frozen_val)

    elif scenario == "sensor_spikes":
        spike_count = max(5, int(n * 0.02))
        spike_indices = np.random.choice(n, size=spike_count, replace=False)
        std_val = np.std(original_values) if np.std(original_values) > 0 else 5.0
        for s_idx in spike_indices:
            direction = 1 if np.random.rand() > 0.5 else -1
            spike_magnitude = direction * (std_val * np.random.uniform(3.5, 6.0))
            corrupted_values[s_idx] += spike_magnitude
            anomaly_mask[s_idx] = True
        meta["spike_indices"] = [int(i) for i in spike_indices]

    elif scenario == "compound":
        # 1. 24h block gap
        block_len = 24
        start_b = n // 5
        missing_mask[start_b : start_b + block_len] = True
        corrupted_values[start_b : start_b + block_len] = np.nan
        
        # 2. 15% random missing outside block
        remaining_indices = np.where(~missing_mask)[0]
        rand_missing = np.random.choice(remaining_indices, size=int(len(remaining_indices) * 0.15), replace=False)
        missing_mask[rand_missing] = True
        corrupted_values[rand_missing] = np.nan

        # 3. Stuck sensor (20 hours)
        valid_indices = np.where(~missing_mask)[0]
        if len(valid_indices) > 50:
            freeze_start = valid_indices[len(valid_indices) // 2]
            freeze_end = min(freeze_start + 20, n)
            corrupted_values[freeze_start:freeze_end] = original_values[freeze_start]
            anomaly_mask[freeze_start + 1 : freeze_end] = True

        # 4. A few sudden spikes
        for _ in range(4):
            sp_idx = np.random.randint(0, n)
            if not missing_mask[sp_idx] and not anomaly_mask[sp_idx]:
                corrupted_values[sp_idx] += (np.std(original_values) + 1.0) * 4.5
                anomaly_mask[sp_idx] = True

    return pd.Series(corrupted_values, index=df.index), missing_mask, anomaly_mask, meta
