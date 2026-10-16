"""
Offline Multi-Variate Training Suite for AI Climate Data Quality Enhancement
Trains Spatio-Temporal Transformer and Bi-LSTM on full 8,760 hours of real climate data
Leverages NVIDIA RTX 4060 GPU with PyTorch CUDA.
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Ensure checkpoints directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def extract_climate_features(df: pd.DataFrame, target_col: str = "temperature"):
    """
    Extracts multi-variate climate and cyclical temporal features.
    Returns:
        feature_matrix: np.ndarray [N, F]
        target_series: np.ndarray [N]
        stats: dict of mean and std for each column
    """
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    
    # Cyclical temporal harmonics
    hour = timestamps.dt.hour.values
    day_of_year = timestamps.dt.dayofyear.values
    
    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Atmospheric multi-variate sensors
    aux_cols = ["solar_radiation", "relative_humidity", "surface_pressure", "wind_speed"]
    stats = {}
    
    # Target stats
    target_vals = df[target_col].values.astype(float)
    stats[target_col] = {"mean": float(np.nanmean(target_vals)), "std": float(np.nanstd(target_vals)) or 1.0}
    norm_target = (target_vals - stats[target_col]["mean"]) / stats[target_col]["std"]
    
    aux_features = []
    for col in aux_cols:
        if col in df.columns:
            vals = df[col].values.astype(float)
            m = float(np.nanmean(vals))
            s = float(np.nanstd(vals)) or 1.0
            stats[col] = {"mean": m, "std": s}
            aux_features.append((vals - m) / s)
        else:
            stats[col] = {"mean": 0.0, "std": 1.0}
            aux_features.append(np.zeros(len(df)))
            
    # Combine: [norm_target, sin_hour, cos_hour, sin_day, cos_day, solar, rh, pressure, wind]
    feat_stack = np.column_stack([
        norm_target,
        sin_hour, cos_hour,
        sin_day, cos_day,
        *aux_features
    ]) # Shape: [N, 9]
    
    return feat_stack.astype(np.float32), target_vals.astype(np.float32), stats

class ClimateImputationDataset(Dataset):
    """
    Sliding window dataset with realistic synthetic missingness injection:
    - 24h block outage
    - 10-40% random MCAR
    - Sensor flatline freeze
    """
    def __init__(self, features: np.ndarray, seq_len: int = 72, stride: int = 6):
        self.features = features
        self.seq_len = seq_len
        self.indices = list(range(0, len(features) - seq_len + 1, stride))
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start = self.indices[idx]
        seq = self.features[start : start + self.seq_len].copy()
        
        target = seq[:, 0].copy() # Normalized true target
        mask = np.ones(self.seq_len, dtype=np.float32) # 1 = observed, 0 = missing
        
        mode = np.random.choice(["block_24", "random", "freeze", "clean"], p=[0.45, 0.35, 0.15, 0.05])
        
        if mode == "block_24":
            gap_len = min(24, self.seq_len // 2)
            gap_start = np.random.randint(4, self.seq_len - gap_len - 2)
            mask[gap_start : gap_start + gap_len] = 0.0
            
        elif mode == "random":
            rate = np.random.uniform(0.10, 0.40)
            drop_idx = np.random.choice(self.seq_len, size=int(self.seq_len * rate), replace=False)
            mask[drop_idx] = 0.0
            
        elif mode == "freeze":
            freeze_len = np.random.randint(12, 24)
            freeze_start = np.random.randint(2, self.seq_len - freeze_len - 1)
            freeze_val = target[freeze_start]
            target[freeze_start : freeze_start + freeze_len] = freeze_val
            mask[freeze_start : freeze_start + freeze_len] = 0.0
            
        # Corrupted input: 0 where missing
        corrupted_target = target * mask
        
        # Input features: [corrupted_target, mask, sin_hour, cos_hour, sin_day, cos_day, solar, rh, pressure, wind]
        x = np.column_stack([corrupted_target, mask, seq[:, 1:]]) # Shape: [seq_len, 10]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

class SpatioTemporalClimateTransformer(nn.Module):
    def __init__(self, input_dim: int = 10, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        h = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        h = self.transformer(h)
        pred = self.head(h).squeeze(-1)
        return pred

class ClimateBiLSTM(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 48, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.lstm(h)
        pred = self.head(out).squeeze(-1)
        return pred

def train_transformer():
    print("\n" + "=" * 60)
    print("[TRAINING] SpatioTemporal Climate Transformer on GPU...")
    print("=" * 60)
    
    # Load full 8,760 hours of ERA5
    data_path = os.path.join(BASE_DIR, "data", "era5_real_data.csv")
    df = pd.read_csv(data_path)
    print(f"[Data] Loaded {len(df)} records from {data_path}")
    
    feat_matrix, target_raw, stats = extract_climate_features(df, target_col="temperature")
    
    # Split train (first 7,000 hours ~ 9.5 months) and val (remaining 1,760 hours ~ 2.5 months)
    train_feat = feat_matrix[:7000]
    val_feat = feat_matrix[7000:]
    
    train_dataset = ClimateImputationDataset(train_feat, seq_len=72, stride=4)
    val_dataset = ClimateImputationDataset(val_feat, seq_len=72, stride=6)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = SpatioTemporalClimateTransformer(input_dim=10, d_model=64, nhead=4, num_layers=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    
    best_val_loss = float("inf")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "transformer_checkpoint.pt")
    
    start_t = time.perf_counter()
    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, y, mask in train_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            
            # Loss heavily weighted on missing tokens
            missing_mask = (1.0 - mask)
            loss_missing = (loss_fn(preds, y) * missing_mask).sum() / (missing_mask.sum() + 1e-6)
            loss_obs = (loss_fn(preds, y) * mask).sum() / (mask.sum() + 1e-6)
            loss = 0.85 * loss_missing + 0.15 * loss_obs
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                preds = model(x)
                missing_mask = (1.0 - mask)
                l = (loss_fn(preds, y) * missing_mask).sum() / (missing_mask.sum() + 1e-6)
                val_loss += l.item()
                val_batches += 1
                
        avg_val = val_loss / max(1, val_batches)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "stats": stats,
                "input_dim": 10,
                "d_model": 64,
                "val_loss": best_val_loss
            }, ckpt_path)
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {total_loss/n_batches:.4f} | Val Missing Loss: {avg_val:.4f}")
            
    elapsed = time.perf_counter() - start_t
    print(f"[DONE] Transformer Training Complete in {elapsed:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path

def train_bilstm():
    print("\n" + "=" * 60)
    print("[TRAINING] Bi-LSTM Imputation Model on GPU...")
    print("=" * 60)
    
    data_path = os.path.join(BASE_DIR, "data", "era5_real_data.csv")
    df = pd.read_csv(data_path)
    feat_matrix, target_raw, stats = extract_climate_features(df, target_col="temperature")
    
    train_dataset = ClimateImputationDataset(feat_matrix[:7000], seq_len=72, stride=4)
    val_dataset = ClimateImputationDataset(feat_matrix[7000:], seq_len=72, stride=6)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = ClimateBiLSTM(input_dim=10, hidden_dim=48, num_layers=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    
    best_val_loss = float("inf")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "bilstm_checkpoint.pt")
    
    start_t = time.perf_counter()
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            missing_mask = (1.0 - mask)
            loss = (loss_fn(preds, y) * missing_mask).sum() / (missing_mask.sum() + 1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                preds = model(x)
                missing_mask = (1.0 - mask)
                val_loss += ((loss_fn(preds, y) * missing_mask).sum() / (missing_mask.sum() + 1e-6)).item()
                val_batches += 1
                
        avg_val = val_loss / max(1, val_batches)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "stats": stats,
                "input_dim": 10,
                "val_loss": best_val_loss
            }, ckpt_path)
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {total_loss/n_batches:.4f} | Val Missing Loss: {avg_val:.4f}")
            
    print(f"[DONE] Bi-LSTM Training Complete in {time.perf_counter() - start_t:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    return ckpt_path

if __name__ == "__main__":
    train_transformer()
    train_bilstm()
