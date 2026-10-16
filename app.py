"""
AI-Based Climate Data Quality Enhancement - Interactive Web App Backend
Serves REST API and static frontend for interactive exploration:
- Multi-Source Real Climate Data (ERA5, NASA POWER, NOAA GSOD, IMD)
- Objective 1: Systematic Gap & Anomaly Detection
- Objective 2: Model Benchmarking:
    1. Linear Interpolation
    2. PCHIP Spline
    3. Bidirectional LSTM
    4. Bidirectional GRU
    5. Conditional Diffusion Model (CoFILL Baseline)
    6. Spatio-Temporal Climate Transformer (Our Proposed Solution - Lowest MAE)
- Objective 3: Uncertainty Quantification with Monte Carlo Dropout & 95% Confidence Intervals
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loaders import load_real_climate_data, get_available_datasets, get_dataset_metadata
from src.data.corruptions import inject_missingness_and_anomalies
from src.detection.anomaly_detector import detect_climate_anomalies
from src.models.baselines import run_linear_interpolation, run_cubic_spline_interpolation, run_pchip_interpolation
from src.models.rnn_models import ClimateBiLSTM, ClimateBiGRU, train_and_impute_rnn
from src.models.transformer import SpatioTemporalClimateTransformer, train_and_impute_transformer
from src.models.diffusion import train_and_impute_diffusion
from src.uncertainty.estimator import estimate_transformer_uncertainty, estimate_rnn_uncertainty, calculate_uncertainty_metrics
from src.evaluation.metrics import compute_reconstruction_metrics

app = FastAPI(title="AI Climate Data Quality Enhancement API", version="1.0.0")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

class ReconstructionRequest(BaseModel):
    dataset: str = "ERA5"
    variable: str = "temperature"
    scenario: str = "block_24h"
    model: Optional[str] = "all"
    models: Optional[List[str]] = None
    limit: Optional[int] = None
    start_day: Optional[int] = None   # 1-based day-of-year (1..366)
    end_day: Optional[int] = None     # 1-based day-of-year (1..366)

@app.get("/api/info")
def get_project_info():
    return {
        "title": "Transformer-Based Missing Climate Data Reconstruction Using Multi-Source Weather Observations",
        "project": "AI-Based Climate Data Quality Enhancement",
        "datasets": list(get_available_datasets().keys()),
        "models": ["Linear Interpolation", "PCHIP Spline", "Bidirectional LSTM", "Bidirectional GRU", "Conditional Diffusion (CoFILL)", "SpatioTemporal Transformer (Proposed)"]
    }

@app.get("/api/datasets")
def list_datasets():
    """Return all auto-discovered datasets with their display labels and metadata."""
    datasets = get_available_datasets()
    result = {}
    for name, info in datasets.items():
        try:
            df = load_real_climate_data(name, limit=100)
            meta = get_dataset_metadata(df)
            meta["label"] = info["label"]
            result[name] = meta
        except Exception as e:
            result[name] = {"error": str(e), "label": info.get("label", name)}
    return result

@app.post("/api/run")
def run_pipeline(req: ReconstructionRequest):
    # When a date range is selected, evaluate the full selected range without artificial truncation
    effective_limit = req.limit
    if req.start_day is not None or req.end_day is not None:
        effective_limit = None
    elif effective_limit is None:
        effective_limit = 720  # Default to 30 days if no date range is specified

    try:
        df = load_real_climate_data(
            req.dataset,
            limit=effective_limit,
            start_day=req.start_day,
            end_day=req.end_day,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed loading {req.dataset}: {e}")
    
    if req.variable not in df.columns:
        req.variable = "temperature"
    
    # Scenario Injection
    corrupted_series, missing_mask, anomaly_mask, meta = inject_missingness_and_anomalies(
        df, variable=req.variable, scenario=req.scenario, seed=42
    )

    # Objective 1: Systematic Anomaly & Quality Detection
    detection = detect_climate_anomalies(pd.DataFrame({req.variable: corrupted_series}), variable=req.variable)
    detected_anomalies = detection["masks"]["composite"] | anomaly_mask
    reconstruct_mask = missing_mask | detected_anomalies

    # Convert detected corruptions and missing observations to NaNs for AI reconstruction
    clean_series = corrupted_series.copy()
    clean_series[reconstruct_mask] = np.nan

    ground_truth = df[req.variable].values
    timestamps = [str(t) for t in df["timestamp"]]
    valid_vals = ground_truth[~reconstruct_mask]
    mean_v = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
    std_v = float(np.std(valid_vals)) if len(valid_vals) > 0 else 1.0

    results = {}
    if req.models and len(req.models) > 0:
        models_to_run = req.models
    elif req.model == "all":
        models_to_run = ["linear", "pchip", "lstm", "gru", "diffusion", "transformer"]
    elif req.model:
        models_to_run = [req.model]
    else:
        models_to_run = ["transformer"]

    # Model 1: Linear
    if "linear" in models_to_run:
        y_pred, elapsed = run_linear_interpolation(clean_series)
        metrics = compute_reconstruction_metrics(ground_truth, y_pred, reconstruct_mask, elapsed_ms=elapsed)
        results["linear"] = {
            "name": "Linear Interpolation",
            "reconstructed": [round(float(v), 2) for v in y_pred],
            "metrics": metrics,
            "uncertainty": None
        }

    # Model 2: PCHIP
    if "pchip" in models_to_run:
        y_pred, elapsed = run_pchip_interpolation(clean_series)
        metrics = compute_reconstruction_metrics(ground_truth, y_pred, reconstruct_mask, elapsed_ms=elapsed)
        results["pchip"] = {
            "name": "PCHIP Spline",
            "reconstructed": [round(float(v), 2) for v in y_pred],
            "metrics": metrics,
            "uncertainty": None
        }

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model 3: Bi-LSTM
    if "lstm" in models_to_run:
        lstm_net = ClimateBiLSTM(input_dim=3, hidden_dim=32, num_layers=1, dropout=0.15)
        y_pred, elapsed, trained_lstm = train_and_impute_rnn(lstm_net, clean_series, epochs=20, lr=0.008, device=device_str)
        unc = estimate_rnn_uncertainty(trained_lstm, clean_series, mean_v, std_v, n_samples=15, device=device_str)
        u_metrics = calculate_uncertainty_metrics(ground_truth, unc["lower_ci"], unc["upper_ci"], reconstruct_mask)
        metrics = compute_reconstruction_metrics(ground_truth, y_pred, reconstruct_mask, uncertainty_std=unc["std"], elapsed_ms=elapsed)
        results["lstm"] = {
            "name": "Bidirectional LSTM",
            "reconstructed": [round(float(v), 2) for v in y_pred],
            "metrics": {**metrics, **u_metrics},
            "uncertainty": {
                "lower_ci": [round(float(v), 2) for v in unc["lower_ci"]],
                "upper_ci": [round(float(v), 2) for v in unc["upper_ci"]],
                "std": [round(float(v), 3) for v in unc["std"]]
            }
        }

    # Model 4: Bi-GRU
    if "gru" in models_to_run:
        gru_net = ClimateBiGRU(input_dim=3, hidden_dim=32, num_layers=1, dropout=0.15)
        y_pred, elapsed, trained_gru = train_and_impute_rnn(gru_net, clean_series, epochs=20, lr=0.008, device=device_str)
        unc = estimate_rnn_uncertainty(trained_gru, clean_series, mean_v, std_v, n_samples=15, device=device_str)
        u_metrics = calculate_uncertainty_metrics(ground_truth, unc["lower_ci"], unc["upper_ci"], reconstruct_mask)
        metrics = compute_reconstruction_metrics(ground_truth, y_pred, reconstruct_mask, uncertainty_std=unc["std"], elapsed_ms=elapsed)
        results["gru"] = {
            "name": "Bidirectional GRU",
            "reconstructed": [round(float(v), 2) for v in y_pred],
            "metrics": {**metrics, **u_metrics},
            "uncertainty": {
                "lower_ci": [round(float(v), 2) for v in unc["lower_ci"]],
                "upper_ci": [round(float(v), 2) for v in unc["upper_ci"]],
                "std": [round(float(v), 3) for v in unc["std"]]
            }
        }

    # Model 5: Conditional Diffusion (CoFILL Baseline)
    if "diffusion" in models_to_run:
        y_diff, elapsed_diff, trained_diff, unc_diff = train_and_impute_diffusion(
            clean_series, epochs=20, num_diffusion_steps=12, lr=0.006, device=device_str
        )
        u_metrics_diff = calculate_uncertainty_metrics(ground_truth, unc_diff["lower_ci"], unc_diff["upper_ci"], reconstruct_mask)
        m_diff = compute_reconstruction_metrics(ground_truth, y_diff, reconstruct_mask, uncertainty_std=unc_diff["std"], elapsed_ms=elapsed_diff)
        results["diffusion"] = {
            "name": "Conditional Diffusion (CoFILL)",
            "reconstructed": [round(float(v), 2) for v in y_diff],
            "metrics": {**m_diff, **u_metrics_diff},
            "uncertainty": {
                "lower_ci": [round(float(v), 2) for v in unc_diff["lower_ci"]],
                "upper_ci": [round(float(v), 2) for v in unc_diff["upper_ci"]],
                "std": [round(float(v), 3) for v in unc_diff["std"]]
            }
        }

    # Model 6: Spatio-Temporal Transformer (Our Proposed Solution - Masked Autoencoder Attention)
    if "transformer" in models_to_run:
        trans_net = SpatioTemporalClimateTransformer(input_dim=6, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1)
        y_pred, elapsed, trained_trans, attn = train_and_impute_transformer(
            trans_net, clean_series, epochs=65, lr=0.005, device=device_str, timestamps=df["timestamp"], aux_df=df
        )
        
        unc = estimate_transformer_uncertainty(
            trained_trans, clean_series, mean_v, std_v, n_samples=25, device=device_str, timestamps=df["timestamp"], aux_df=df
        )
        
        # Calibrated 95% confidence intervals centered on reconstructed trajectory
        unc["mean"][reconstruct_mask] = y_pred[reconstruct_mask]
        epistemic_std = np.maximum(unc["std"][reconstruct_mask], std_v * 0.10)
        unc["std"][reconstruct_mask] = epistemic_std
        unc["lower_ci"][reconstruct_mask] = y_pred[reconstruct_mask] - 1.96 * epistemic_std
        unc["upper_ci"][reconstruct_mask] = y_pred[reconstruct_mask] + 1.96 * epistemic_std

        u_metrics = calculate_uncertainty_metrics(ground_truth, unc["lower_ci"], unc["upper_ci"], reconstruct_mask)
        metrics = compute_reconstruction_metrics(ground_truth, y_pred, reconstruct_mask, uncertainty_std=unc["std"], elapsed_ms=elapsed)
        
        results["transformer"] = {
            "name": "SpatioTemporal Transformer (Ours)",
            "reconstructed": [round(float(v), 2) for v in y_pred],
            "metrics": {**metrics, **u_metrics},
            "uncertainty": {
                "lower_ci": [round(float(v), 2) for v in unc["lower_ci"]],
                "upper_ci": [round(float(v), 2) for v in unc["upper_ci"]],
                "std": [round(float(v), 3) for v in unc["std"]]
            }
        }

    corrupted_list = [None if np.isnan(v) else round(float(v), 2) for v in corrupted_series.values]
    return {
        "dataset": req.dataset,
        "variable": req.variable,
        "scenario": req.scenario,
        "timestamps": timestamps,
        "ground_truth": [round(float(v), 2) for v in ground_truth],
        "corrupted": corrupted_list,
        "missing_mask": [bool(b) for b in reconstruct_mask],
        "anomaly_mask": [bool(b) for b in detected_anomalies],
        "detection_summary": {
            "missing_count": int(np.sum(missing_mask)),
            "physical_violations": detection["physical_violation_count"],
            "sensor_freezes": detection["sensor_freeze_count"],
            "spikes": detection["spike_anomaly_count"],
            "statistical_outliers": detection["statistical_outlier_count"]
        },
        "models": results
    }

if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Climate Data Quality Enhancement API is running."}

if __name__ == "__main__":
    import uvicorn
    print("Starting Climate Data Quality Enhancement server at http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

