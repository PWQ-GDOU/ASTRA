"""
Complete SHRDL Pretraining Script — Multi-Source Degradation Data
Inputs: Nozzle ablation + C-MAPSS turbofan + NASA Battery
Output: Pre-trained SHRDL ADN encoder
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import argparse, json, glob
from datetime import datetime

from src.config import SHRDLConfig, CHECKPOINT_DIR
from src.models.shrdl import SHRDL, CIMCLLoss
from src.data.preprocess import extract_sliding_windows
from src.utils.train_utils import set_seed


def load_cmapss_trajectories(data_dir: str, seq_len: int) -> list:
    """Load all C-MAPSS trajectories from txt files."""
    trajectories = []
    cmapss_dir = os.path.join(data_dir, 'cmapss', '6. Turbofan Engine Degradation Simulation Data Set')
    
    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        train_file = os.path.join(cmapss_dir, f'train_{fd}.txt')
        if not os.path.exists(train_file):
            continue
        
        try:
            # Columns: unit, time, op1, op2, op3, sensor1..sensor21
            df = pd.read_csv(train_file, sep=r'\s+', header=None)
            data = df.values
            
            # Group by unit (column 0)
            units = np.unique(data[:, 0])
            for unit in units:
                mask = data[:, 0] == unit
                traj = data[mask, 2:]  # Skip unit_id and time columns
                if len(traj) >= seq_len:
                    trajectories.append(traj.astype(np.float32))
            print(f"  C-MAPSS {fd}: {len(units)} units")
        except Exception as e:
            print(f"  Warning: C-MAPSS {fd} error: {e}")
    
    return trajectories


def load_battery_trajectories(data_dir: str, seq_len: int) -> list:
    """Load NASA battery degradation trajectories from .mat files."""
    trajectories = []
    battery_dir = os.path.join(data_dir, 'nasa_battery', '5. Battery Data Set')
    
    mat_files = sorted(glob.glob(os.path.join(battery_dir, 'B*.mat')))
    for mat_file in mat_files:
        try:
            mat = loadmat(mat_file)
            batt = mat[list(mat.keys())[-1]]  # e.g., 'B0005'
            
            # Battery data structure: each row is a charge/discharge cycle
            # Extract relevant features: voltage, current, temperature, capacity
            features = []
            for cycle_idx in range(batt.shape[0]):
                cycle = batt[cycle_idx, 0]
                if cycle is not None and len(cycle) > 0:
                    if isinstance(cycle, np.ndarray) and cycle.ndim > 1:
                        # Extract scalar features per cycle:
                        # - discharge capacity / rated capacity (SOH proxy)
                        # We need features that change over cycles
                        row_data = []
                        for col in range(min(cycle.shape[1], 10)):
                            vals = cycle[:, col]
                            row_data.append(float(np.nanmean(vals)))
                        features.append(row_data)
            
            if len(features) >= seq_len:
                traj = np.array(features, dtype=np.float32)
                trajectories.append(traj)
        except Exception as e:
            pass  # Skip problematic files
    
    print(f"  Battery: {len(trajectories)} batteries loaded")
    return trajectories


def load_nozzle_trajectories(data_dir: str, seq_len: int) -> list:
    """Load nozzle ablation data."""
    trajectories = []
    raw_dir = os.path.join(data_dir, 'raw')
    
    for fname in ['喷管喉部退化时程.csv', '壁面烧蚀速率分布数据.csv']:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, encoding='utf-8-sig')
                data = df.values.astype(np.float32)
                if len(data) >= seq_len:
                    trajectories.append(data)
                print(f"  Nozzle {fname}: {len(data)} timesteps")
            except Exception as e:
                print(f"  Warning: {fname}: {e}")
    
    return trajectories


def prepare_windows(trajectories: list, seq_len: int, stride: int) -> np.ndarray:
    """Extract sliding windows from all trajectories and normalize."""
    all_windows = []
    for traj in trajectories:
        if traj.shape[1] < 2:
            continue
        # Handle NaN/Inf
        traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        
        X, _ = extract_sliding_windows(traj, seq_len, stride)
        if len(X) > 0:
            # Per-trajectory normalization
            mean = X.mean(axis=(0, 1), keepdims=True)
            std = X.std(axis=(0, 1), keepdims=True) + 1e-8
            X = (X - mean) / std
            all_windows.append(X)
    
    if not all_windows:
        return np.array([])
    
    return np.concatenate(all_windows, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = f"cuda:{args.gpu}"
    
    config = SHRDLConfig()
    config.seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.epochs = args.epochs
    
    print(f"{'='*60}")
    print(f"  SHRDL Pretraining — GPU{args.gpu} Seed{args.seed}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # ========== Load Data ==========
    print("\n[1/4] Loading multi-source data...")
    
    all_traj = []
    all_traj += load_nozzle_trajectories(args.data_dir, args.seq_len)
    all_traj += load_cmapss_trajectories(args.data_dir, args.seq_len)
    all_traj += load_battery_trajectories(args.data_dir, args.seq_len)
    
    print(f"  Total trajectories: {len(all_traj)}")
    
    # ========== Prepare Windows ==========
    print("\n[2/4] Extracting sliding windows...")
    X_all = prepare_windows(all_traj, args.seq_len, args.stride)
    print(f"  Total windows: {len(X_all)} x {args.seq_len} x {X_all.shape[-1] if len(X_all) > 0 else '?'}")
    
    if len(X_all) < args.batch_size * 2:
        print("  ERROR: Not enough data! Need more trajectories.")
        return
    
    # Pad features to consistent dimension (use max across all trajectories)
    n_features = X_all.shape[-1]
    
    # ========== Build Model ==========
    print(f"\n[3/4] Building SHRDL (n_features={n_features})...")
    model = SHRDL(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_adn_blocks=config.n_adn_blocks,
        n_features=n_features,
        feature_dim=config.feature_dim,
        window_sizes=config.ndl_window_sizes,
        momentum=config.momentum,
        memory_bank_size=config.memory_bank_size,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CIMCLLoss(temperature=config.temperature)
    
    # ========== Train ==========
    print(f"\n[4/4] Training ({args.epochs} epochs, {args.batch_size}/batch)...")
    best_loss = float('inf')
    n_samples = len(X_all)
    n_batches = n_samples // args.batch_size
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle each epoch
        indices = np.random.permutation(n_samples)
        
        pbar = tqdm(range(n_batches), desc=f"GPU{args.gpu} E{epoch}")
        for batch_idx in pbar:
            start = (batch_idx * args.batch_size) % n_samples
            idx = indices[start:start + args.batch_size]
            
            x_batch = torch.FloatTensor(X_all[idx]).to(device)
            
            # Create two views via simple noise (lightweight augmentation)
            noise1 = 0.01 * torch.randn_like(x_batch)
            noise2 = 0.01 * torch.randn_like(x_batch)
            view1 = x_batch + noise1
            view2 = x_batch + noise2
            
            # Forward: contrastive learning
            q, k1, k2 = model(view1, view2)
            
            # CIMCL loss with memory queue
            queue = model.memory_bank.get_queue().to(device)
            loss = criterion(q, k2, queue)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "queue": f"{len(queue)}"})
        
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  GPU{args.gpu} Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(CHECKPOINT_DIR, f"shrdl_gpu{args.gpu}_best.pt")
            torch.save({
                "epoch": epoch, "loss": best_loss,
                "model": model.state_dict(),
                "config": config, "n_features": n_features,
            }, save_path)
    
    print(f"\n  GPU{args.gpu} DONE! Best loss: {best_loss:.4f}")
    print(f"  Model saved: {save_path}")


if __name__ == "__main__":
    main()
