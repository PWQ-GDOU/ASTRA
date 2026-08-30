"""
Server Training Script — SHRDL Pretraining on GPU 1-3 in parallel.
Launches 3 independent pretraining runs with different seeds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from src.config import SHRDLConfig, CHECKPOINT_DIR
from src.models.shrdl import SHRDL, CIMCLLoss
from src.data.preprocess import load_nozzle_data, extract_sliding_windows
from src.utils.train_utils import set_seed, EarlyStopping, AverageMeter


def load_nozzle_trajectories(data_dir: str, seq_len: int) -> list:
    """Load and prepare nozzle ablation data for pretraining."""
    nozzle = load_nozzle_data(data_dir)
    throat = nozzle["throat_degradation"]
    
    # Use all feature columns (skip header row / BOM row if present)
    # Columns: time, ablation_depth, throat_radius, ablation_rate, mach_in, mach_out
    if throat.shape[1] <= 2:
        return []
    
    # Extract as single trajectory    
    return [throat]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)  # Small dataset
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = f"cuda:{args.gpu}"
    
    config = SHRDLConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.seq_len = 32  # Must be < nozzle data length (51 timesteps)
    
    print(f"{'='*60}")
    print(f"  SHRDL Pretraining — GPU {args.gpu} — Seed {args.seed}")
    print(f"{'='*60}")
    
    # Load data
    trajectories = load_nozzle_trajectories(args.data_dir, config.seq_len)
    if not trajectories:
        print("ERROR: No nozzle data found!")
        return
    
    # Extract windows for contrastive learning
    all_windows = []
    for traj in trajectories:
        X, _ = extract_sliding_windows(traj, config.seq_len, stride=2)
        if len(X) > 0:
            # Normalize
            X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-8)
            all_windows.append(X)
    
    if not all_windows:
        print("ERROR: No valid windows!")
        return
    
    X_all = np.concatenate(all_windows, axis=0).astype(np.float32)
    print(f"  Dataset: {len(X_all)} windows of length {config.seq_len}")
    
    # Create model
    n_features = X_all.shape[-1]
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
    
    # Training loop
    best_loss = float('inf')
    n_batches = len(X_all) // args.batch_size
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle for each epoch
        indices = np.random.permutation(len(X_all))
        
        pbar = tqdm(range(n_batches), desc=f"GPU{args.gpu} E{epoch}/{args.epochs}")
        for batch_idx in pbar:
            start = (batch_idx * args.batch_size) % len(X_all)
            idx = indices[start:start + args.batch_size]
            
            x_batch = torch.FloatTensor(X_all[idx]).to(device)
            
            # Create two augmented views
            noise1 = 0.01 * torch.randn_like(x_batch)
            noise2 = 0.01 * torch.randn_like(x_batch)
            view1 = x_batch + noise1
            view2 = x_batch + noise2
            
            # Forward
            q, k1, k2 = model(view1, view2)
            
            # CIMCL loss
            queue = model.memory_bank.get_queue().to(device)
            loss = criterion(q, k2, queue)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  GPU{args.gpu} Epoch {epoch}: avg_loss={avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                CHECKPOINT_DIR, f"shrdl_gpu{args.gpu}_seed{args.seed}_best.pt"
            )
            torch.save({
                "epoch": epoch, "loss": best_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }, save_path)
            print(f"  → Best checkpoint saved: {save_path}")
    
    print(f"\n  GPU{args.gpu} complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
