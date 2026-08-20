"""
SHRDL Pretraining Script — Stage 1 of the pipeline.

Pre-trains the ADN encoder on multi-source degradation data
using CIMCL contrastive learning.

Usage:
    python scripts/pretrain_shrdl.py --data_dir data/processed --epochs 200 --batch_size 128
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from src.config import SHRDLConfig, ROOT_DIR, CHECKPOINT_DIR
from src.models.shrdl import SHRDL, CIMCLLoss
from src.data.dataset import ContrastiveDegradationDataset, create_dataloaders
from src.data.preprocess import load_nozzle_data, load_cmapss, load_battery_data
from src.utils.train_utils import set_seed, EarlyStopping, AverageMeter


def prepare_multi_source_data(config: SHRDLConfig, nozzle_dir: str, 
                               public_dir: str) -> list:
    """Load and combine all source domain datasets for pretraining."""
    all_trajectories = []
    
    # 1. Nozzle ablation data (user's own)
    try:
        nozzle = load_nozzle_data(nozzle_dir)
        throat = nozzle["throat_degradation"]
        if len(throat) > 0:
            all_trajectories.append(throat)
            print(f"  Loaded nozzle ablation: {len(throat)} timesteps")
    except Exception as e:
        print(f"  Warning: Nozzle data not available: {e}")
    
    # 2. C-MAPSS data
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            cmapss = load_cmapss(public_dir, subset)
            if cmapss.get("data") is not None:
                # Process C-MAPSS trajectories (each unit is a trajectory)
                raw = cmapss["data"]
                # Columns: unit, time, op1, op2, op3, sensor1..sensor21
                # Group by unit
                units = np.unique(raw[:, 0])
                for unit in units:
                    mask = raw[:, 0] == unit
                    traj = raw[mask, 2:]  # skip unit and time columns
                    if len(traj) > config.seq_len:
                        all_trajectories.append(traj)
                print(f"  Loaded C-MAPSS {subset}: {len(units)} units")
        except Exception as e:
            print(f"  Warning: C-MAPSS {subset} not available: {e}")
    
    # 3. Battery data
    for dataset_name in ["nasa", "calce"]:
        try:
            battery = load_battery_data(public_dir, dataset_name)
            cycles = battery.get("cycles", {})
            for batt_id, cycle_data in cycles.items():
                if len(cycle_data) > config.seq_len:
                    all_trajectories.append(cycle_data)
            print(f"  Loaded {dataset_name} battery: {len(cycles)} batteries")
        except Exception as e:
            print(f"  Warning: {dataset_name} battery not available: {e}")
    
    print(f"\n  Total trajectories: {len(all_trajectories)}")
    return all_trajectories


def train_epoch(model, dataloader, criterion, optimizer, epoch, config):
    """One epoch of contrastive pretraining."""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for view1, view2 in pbar:
        view1 = view1.to(config.device)
        view2 = view2.to(config.device)
        
        # Forward: get query and key features
        q, k1, k2 = model(view1, view2)
        
        # CIMCL loss
        queue = model.memory_bank.get_queue().to(config.device)
        loss = criterion(q, k2, queue)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.update(loss.item(), view1.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    return losses.avg


def main():
    parser = argparse.ArgumentParser(description="SHRDL Pretraining")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Processed data directory")
    parser.add_argument("--nozzle_dir", type=str, default=r"D:\数据集\航天器",
                       help="Nozzle ablation data directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    set_seed(42)
    config = SHRDLConfig()
    config.device = args.device
    
    print("="*60)
    print("  SHRDL: Self-Supervised Pretraining")
    print("="*60)
    
    # Prepare data
    print("\n[1/4] Loading multi-source data...")
    trajectories = prepare_multi_source_data(config, args.nozzle_dir, args.data_dir)
    
    if not trajectories:
        print("ERROR: No data found for pretraining!")
        return
    
    dataset = ContrastiveDegradationDataset(
        trajectories,
        seq_len=config.seq_len,
        stride=4,
        normalize=True,
        crop_min=config.crop_min_ratio,
        crop_max=config.crop_max_ratio,
        mask_ratio=config.mask_ratio,
    )
    dataloader = create_dataloaders(dataset, config.batch_size, shuffle=True)
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # Create model
    print(f"\n[2/4] Building SHRDL model...")
    n_features = trajectories[0].shape[1] if len(trajectories) > 0 else 21
    model = SHRDL(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_adn_blocks=config.n_adn_blocks,
        n_features=n_features,
        feature_dim=config.feature_dim,
        window_sizes=config.ndl_window_sizes,
        momentum=config.momentum,
        memory_bank_size=config.memory_bank_size,
    ).to(config.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    print(f"\n[3/4] Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = CIMCLLoss(
        temperature=config.temperature,
        beta=config.cimcl_beta,
        gamma=config.cimcl_gamma,
    )
    early_stopping = EarlyStopping(patience=30)
    
    # Train
    print(f"\n[4/4] Training ({config.epochs} epochs)...")
    best_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, epoch, config)
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "shrdl_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": config,
            }, checkpoint_path)
            print(f"  → Saved best checkpoint (loss={best_loss:.4f})")
        
        if early_stopping(train_loss, epoch):
            break
    
    # Save final encoder
    encoder_path = os.path.join(CHECKPOINT_DIR, "shrdl_encoder.pt")
    torch.save(model.state_dict(), encoder_path)
    print(f"\nPretraining complete! Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {encoder_path}")


if __name__ == "__main__":
    main()
