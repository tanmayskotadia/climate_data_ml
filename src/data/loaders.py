"""
Multi-Source Real Climate Data Loaders
Loads real datasets:
- ERA5 (Hourly Reanalysis, 8760 points)
- NASA POWER (Daily Satellite & Agroclimatology, 365 points)
- NOAA GSOD (Daily Station Observations, 363 points)
- IMD (Ground Station Observations)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

def get_available_datasets() -> Dict[str, str]:
    files = {
        "ERA5": os.path.join(DATA_DIR, "era5_real_data.csv"),
        "NASA_POWER": os.path.join(DATA_DIR, "nasa_power_real_data.csv"),
        "NOAA_GSOD": os.path.join(DATA_DIR, "noaa_gsod_real_data.csv"),
        "IMD": os.path.join(DATA_DIR, "imd_real_data.csv"),
    }
    return {k: v for k, v in files.items() if os.path.exists(v)}

def load_real_climate_data(source: str = "ERA5", limit: Optional[int] = None) -> pd.DataFrame:
    """
    Loads and standardizes real climate dataset.
    Args:
        source: 'ERA5', 'NASA_POWER', 'NOAA_GSOD', or 'IMD'
        limit: Optional maximum number of rows to return (useful for fast testing)
    Returns:
        pd.DataFrame with columns: timestamp, temperature, relative_humidity, wind_speed, precipitation, etc.
    """
    datasets = get_available_datasets()
    if source not in datasets:
        # Fallback to whatever is available
        source = list(datasets.keys())[0]
    
    filepath = datasets[source]
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Clean non-finite values if any
    for col in ["temperature", "relative_humidity", "wind_speed", "precipitation"]:
        if col in df.columns:
            # Fill small internal gaps in baseline with forward/backward fill so we have a clean ground truth
            df[col] = df[col].interpolate(method="linear").bfill().ffill()
    
    if limit is not None and limit < len(df):
        df = df.iloc[:limit].copy()
        
    return df

def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, any]:
    """
    Returns summary statistics and time range for the loaded dataset.
    """
    numeric_cols = [c for c in ["temperature", "relative_humidity", "wind_speed", "precipitation", "solar_radiation"] if c in df.columns]
    stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            stats[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
            }
    
    return {
        "source": df["source"].iloc[0] if "source" in df.columns else "Unknown",
        "total_records": len(df),
        "start_time": str(df["timestamp"].min()),
        "end_time": str(df["timestamp"].max()),
        "variables": numeric_cols,
        "statistics": stats
    }
