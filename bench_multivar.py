import os
import sys
import numpy as np
import pandas as pd
import torch

from src.data.loaders import load_real_climate_data
from src.data.corruptions import inject_missingness_and_anomalies
from src.models.baselines import run_pchip_interpolation, run_linear_interpolation
from src.models.transformer import SpatioTemporalClimateTransformer, train_and_impute_transformer
from src.evaluation.metrics import compute_reconstruction_metrics
from src.uncertainty.estimator import estimate_transformer_uncertainty, calculate_uncertainty_metrics

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device_str} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

df = load_real_climate_data("ERA5", limit=720) # 30 days
scenarios = ["block_24h", "random_10", "random_20", "random_40", "sensor_freeze", "sensor_spikes", "compound"]
variables = ["temperature", "relative_humidity", "wind_speed"]

rows = []
for var in variables:
    for sc in scenarios:
        corr, mask, anom, _ = inject_missingness_and_anomalies(df, variable=var, scenario=sc, seed=42)
        gt = df[var].values
        valid_vals = gt[~mask]
        mean_v = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
        std_v = float(np.std(valid_vals)) if len(valid_vals) > 0 else 1.0

        # Linear
        yl, tl = run_linear_interpolation(corr)
        ml = compute_reconstruction_metrics(gt, yl, mask, elapsed_ms=tl)
        
        # PCHIP
        yp, tp = run_pchip_interpolation(corr)
        mp = compute_reconstruction_metrics(gt, yp, mask, elapsed_ms=tp)
        
        # Transformer
        trans_net = SpatioTemporalClimateTransformer(input_dim=6, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1)
        yt, tt, trained_trans, attn = train_and_impute_transformer(
            trans_net, pd.Series(corr), epochs=35, lr=0.005, device=device_str, timestamps=df["timestamp"], aux_df=df
        )
        unc = estimate_transformer_uncertainty(
            trained_trans, pd.Series(corr), mean_v, std_v, n_samples=20, device=device_str, timestamps=df["timestamp"], aux_df=df
        )
        unc["mean"][mask] = yt[mask]
        ep_std = np.maximum(unc["std"][mask], std_v * 0.10)
        unc["std"][mask] = ep_std
        unc["lower_ci"][mask] = yt[mask] - 1.96 * ep_std
        unc["upper_ci"][mask] = yt[mask] + 1.96 * ep_std
        u_metrics = calculate_uncertainty_metrics(gt, unc["lower_ci"], unc["upper_ci"], mask)

        mt = compute_reconstruction_metrics(gt, yt, mask, elapsed_ms=tt)
        
        rows.append({
            "Variable": var,
            "Scenario": sc,
            "Linear_MAE": round(ml["MAE"], 3),
            "Linear_R2": round(ml.get("R2", -99.0), 4),
            "PCHIP_MAE": round(mp["MAE"], 3),
            "PCHIP_R2": round(mp.get("R2", -99.0), 4),
            "Trans_MAE": round(mt["MAE"], 3),
            "Trans_RMSE": round(mt["RMSE"], 3),
            "Trans_R2": round(mt.get("R2", -99.0), 4),
            "Trans_Lat_ms": round(mt["Latency_ms"], 1),
            "Trans_PICP": round(u_metrics.get("PICP", 0.0), 1),
            "Trans_MPIW": round(u_metrics.get("MPIW", 0.0), 3),
        })
        print(f"Done: {var:<18} | {sc:<14} | Trans MAE: {mt['MAE']:.3f} | PCHIP MAE: {mp['MAE']:.3f} | Trans Latency: {mt['Latency_ms']:.1f}ms")

res_df = pd.DataFrame(rows)
print("\n=== COMPREHENSIVE MULTI-VARIABLE MULTI-SCENARIO BENCHMARK RESULTS ===")
print(res_df.to_string(index=False))
res_df.to_csv("results/multivar_benchmark_results.csv", index=False)
