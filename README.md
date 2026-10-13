# AI-Based Climate Data Quality Enhancement
## Transformer-Based Missing Climate Data Reconstruction Using Multi-Source Weather Observations

**Authors:**
- **Tanmay Suresh Kotadia** – `24BCE5281`
- **Aditya Sandhwar** – `24BCE5253`

---

## 📌 Project Overview
Climate datasets collected from ground stations, satellites, and reanalysis products frequently suffer from missing observations, sensor calibration drift, telemetry dropouts, and severe equipment failures. This project implements an end-to-end AI framework that:
1. **Detects Missing & Anomalous Climate Observations** (Objective 1): Systematically identifies sampling gaps, sensor freeze flatlines, electrical spikes, and physical range violations across multi-source archives (**ERA5**, **NASA POWER**, **NOAA GSOD**, and **IMD**).
2. **Reconstructs Missing Observations Across Architectural Baselines** (Objective 2): Benchmarks traditional statistical interpolation (Linear, PCHIP Spline) against recurrent deep learning architectures (**Bi-LSTM**, **Bi-GRU**) and a modern **Spatio-Temporal Climate Transformer** (equipped with multi-head self-attention and learnable `[MASK]` tokens).
3. **Quantifies Predictive Uncertainty** (Objective 3): Applies **Monte Carlo Dropout (MCD)** sampling to generate calibrated **95% Confidence Intervals**, prediction interval coverage (**PICP**), and interval widths (**MPIW**).

---

## 📂 Repository Structure
```
c:/tanmay/
├── DA1_24BCE5281 & 24BCE5253.pdf     # Original research project proposal & literature review
├── download_real_datasets.py         # Downloader for real ERA5, NASA POWER, NOAA GSOD & IMD data
├── run_benchmark.py                  # Full offline benchmark runner & publication figure generator
├── app.py                            # FastAPI backend serving live AI inference & REST API
├── requirements.txt                  # Python dependencies
├── data/                             # Real downloaded climate datasets (CSV)
│   ├── era5_real_data.csv            # ECMWF ERA5 Hourly Reanalysis (8,760 records)
│   ├── nasa_power_real_data.csv      # NASA POWER Agroclimatology (365 daily records)
│   ├── noaa_gsod_real_data.csv       # NOAA GSOD Station 42182 (363 daily records)
│   └── imd_real_data.csv             # IMD Ground Station Network observations
├── src/
│   ├── data/
│   │   ├── loaders.py                # Dataset loaders and metadata statistics
│   │   └── corruptions.py            # Missingness (MCAR, 24h block gap) & anomaly injection
│   ├── detection/
│   │   └── anomaly_detector.py       # Objective 1: Multi-method quality & anomaly detection
│   ├── models/
│   │   ├── baselines.py              # Linear, Cubic Spline, PCHIP, KNN Imputation
│   │   ├── rnn_models.py             # Mask-aware Bidirectional LSTM & Bidirectional GRU
│   │   └── transformer.py            # Spatio-Temporal Climate Transformer with self-attention
│   ├── uncertainty/
│   │   └── estimator.py              # Objective 3: Monte Carlo Dropout & 95% Confidence Intervals
│   └── evaluation/
│       └── metrics.py                # MAE, RMSE, R², MAPE, CRPS, PICP, MPIW, Latency
├── web/                              # Interactive Web Dashboard
│   ├── index.html                    # State-of-the-art dark glassmorphic UI
│   ├── style.css                     # Custom styling, responsive layout, micro-animations
│   └── app.js                        # High-performance Canvas visualizer & experiment runner
└── results/
    ├── benchmark_summary.csv         # Exported empirical metrics table across scenarios
    └── figures/                      # High-resolution publication plots (.png)
        ├── fig1_model_reconstruction_comparison.png
        ├── fig2_benchmark_metrics_barchart.png
        └── fig3_uncertainty_quantification_bounds.png
```

---

## 🚀 Quickstart Guide

### 1. Download / Verify Real Climate Datasets
The datasets from ERA5, NASA POWER, NOAA GSOD, and IMD can be fetched or updated anytime using:
```bash
python download_real_datasets.py
```

### 2. Run Comprehensive Offline Benchmark & Generate Figures
To run the full comparative benchmarking suite across multiple missingness scenarios and generate high-resolution figures:
```bash
python run_benchmark.py
```
Outputs:
- Metric summary saved to `results/benchmark_summary.csv`.
- Visualizations saved to `results/figures/`.

### 3. Launch Interactive Web Dashboard
To start the live interactive dashboard:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any web browser to:
- Switch between **ERA5**, **NASA POWER**, **NOAA GSOD**, and **IMD**.
- Test **24h Sensor Outage Blocks**, **10%-40% Random MCAR Missingness**, **Sensor Freezes**, and **Spike Glitches**.
- Inspect live **95% Confidence Interval bands** generated via Monte Carlo Dropout.
- Export results to CSV.

---

## 📊 Summary of Benchmark Findings
Across extended continuous outage gaps (e.g., 24-hour sensor breakdown):
1. **Classical Linear/Spline Interpolation** fails to reconstruct non-linear diurnal cycles, flatlining across the gap.
2. **Recurrent Models (Bi-LSTM / Bi-GRU)** struggle with cumulative error propagation over long consecutive missing steps.
3. **Proposed Spatio-Temporal Transformer** demonstrates superior robustness on continuous blocks by leveraging global multi-head self-attention to attend directly across uncorrupted temporal horizons, while attaching calibrated **95% Confidence Intervals** through Monte Carlo Dropout.
