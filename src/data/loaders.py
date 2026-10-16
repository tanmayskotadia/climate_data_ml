"""
Multi-Source Real Climate Data Loaders
Dynamically discovers and loads meteorological datasets from data/. Supports:
  - CSV files with standard schema (timestamp, temperature, relative_humidity, ...)
  - XLSX files with ISHRAE-style columns (Year, Month, Date, Time, DBT, RH, GHR, ...)
  - Auto-discovered datasets from the data/ directory
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# ── Well-known aliases: file-stem → display name ────────────────────────────
_KNOWN_LABELS = {
    "era5_real_data":        "ECMWF ERA5 (Hourly Reanalysis - 8760 pts)",
    "nasa_power_real_data":  "NASA POWER (Agroclimatology - 365 pts)",
    "noaa_gsod_real_data":   "NOAA GSOD (Ground Station - 363 pts)",
    "imd_real_data":         "IMD Network (Mumbai Station)",
    "ishrae2014_real_data":  "ISHRAE 2014 TMY (Chennai Hourly - 8760 pts)",
}

# ── ISHRAE / TMY column mapping ──────────────────────────────────────────────
_ISHRAE_COL_MAP = {
    "DBT":        "temperature",
    "RH":         "relative_humidity",
    "GHR":        "solar_radiation",
    "DNR":        "direct_normal_radiation",
    "Diffuse HR": "diffuse_radiation",
}


def _make_label(stem: str) -> str:
    if stem in _KNOWN_LABELS:
        return _KNOWN_LABELS[stem]
    label = re.sub(r"[._-]+", " ", stem).strip()
    return label


def _convert_xlsx_to_csv(xlsx_path: str) -> str:
    """Convert an ISHRAE-style XLSX to a sidecar CSV on first access."""
    raw = pd.read_excel(xlsx_path)
    raw.columns = [c.strip() for c in raw.columns]

    year_col  = next((c for c in raw.columns if c.lower() == "year"),  None)
    month_col = next((c for c in raw.columns if c.lower() == "month"), None)
    date_col  = next((c for c in raw.columns if c.lower() in ("date", "day")), None)
    time_col  = next((c for c in raw.columns if c.lower() == "time"),  None)

    if not all([year_col, month_col, date_col, time_col]):
        raise ValueError(f"XLSX {xlsx_path} missing Year/Month/Date/Time columns.")

    hour = (raw[time_col].astype(int) - 1).clip(0, 23)
    timestamps = pd.to_datetime({
        "year":  raw[year_col].astype(int),
        "month": raw[month_col].astype(int),
        "day":   raw[date_col].astype(int),
        "hour":  hour,
    })

    out = pd.DataFrame({"timestamp": timestamps})
    for src_col, dst_col in _ISHRAE_COL_MAP.items():
        if src_col in raw.columns:
            out[dst_col] = pd.to_numeric(raw[src_col], errors="coerce")

    stem = os.path.splitext(os.path.basename(xlsx_path))[0]
    out["source"] = _make_label(stem)
    out = out.sort_values("timestamp").reset_index(drop=True)

    csv_path = os.path.splitext(xlsx_path)[0] + "_auto.csv"
    out.to_csv(csv_path, index=False)
    return csv_path


def get_available_datasets() -> Dict[str, Dict]:
    """
    Scan DATA_DIR and return every loadable dataset as:
      key -> {"path": str, "label": str}
    Drop any CSV/XLSX in data/ and it appears here automatically.
    """
    result: Dict[str, Dict] = {}

    if not os.path.isdir(DATA_DIR):
        return result

    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        stem, ext = os.path.splitext(fname)

        # Skip auto-generated sidecar CSVs
        if stem.endswith("_auto"):
            continue

        if ext.lower() == ".csv":
            try:
                header = pd.read_csv(fpath, nrows=0)
                if "timestamp" not in header.columns:
                    continue
            except Exception:
                continue
            key = stem.upper().replace("-", "_")
            result[key] = {"path": fpath, "label": _make_label(stem)}

        elif ext.lower() in (".xlsx", ".xls"):
            sidecar = os.path.splitext(fpath)[0] + "_auto.csv"
            if not os.path.exists(sidecar):
                try:
                    _convert_xlsx_to_csv(fpath)
                except Exception:
                    continue
            key = stem.upper().replace("-", "_").replace(".", "_")
            result[key] = {"path": sidecar, "label": _make_label(stem)}

    return result


def load_real_climate_data(
    source: str = "ERA5_REAL_DATA",
    limit: Optional[int] = None,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and standardise a climate dataset by its auto-generated key.

    Args:
        source:    Dataset key from get_available_datasets().
        limit:     Max rows to return (applied after day slicing).
        start_day: 1-based day-of-year to start from (inclusive). None = beginning.
        end_day:   1-based day-of-year to end at (inclusive). None = end.
    """
    datasets = get_available_datasets()

    # Backwards-compat: accept old short names (ERA5, NASA_POWER, ISHRAE2014 …)
    if source not in datasets:
        for k in datasets:
            if k.startswith(source.upper().replace("-", "_")):
                source = k
                break
        else:
            source = list(datasets.keys())[0]

    filepath = datasets[source]["path"]
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Day-of-year slicing ──────────────────────────────────────────────────
    if start_day is not None or end_day is not None:
        doy = df["timestamp"].dt.dayofyear
        lo  = int(start_day) if start_day is not None else 1
        hi  = int(end_day)   if end_day   is not None else int(doy.max())
        df  = df[(doy >= lo) & (doy <= hi)].reset_index(drop=True)

    # ── Fill baseline column gaps ────────────────────────────────────────────
    for col in ["temperature", "relative_humidity", "wind_speed", "precipitation"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").bfill().ffill()

    if limit is not None and limit < len(df):
        df = df.iloc[:limit].copy()

    return df


def get_dataset_metadata(df: pd.DataFrame) -> Dict:
    """Return summary statistics and time range for a loaded dataset."""
    numeric_cols = [c for c in [
        "temperature", "relative_humidity", "wind_speed",
        "precipitation", "solar_radiation",
    ] if c in df.columns]

    stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            stats[col] = {
                "mean": round(float(series.mean()), 2),
                "std":  round(float(series.std()),  2),
                "min":  round(float(series.min()),  2),
                "max":  round(float(series.max()),  2),
            }

    return {
        "source":        df["source"].iloc[0] if "source" in df.columns else "Unknown",
        "total_records": len(df),
        "start_time":    str(df["timestamp"].min()),
        "end_time":      str(df["timestamp"].max()),
        "variables":     numeric_cols,
        "statistics":    stats,
    }
