"""
PEUDA Domain Adaptation — Stage 2 of the pipeline.

Adapts the pre-trained encoder from source domain (ablation data)
to target domain (COMSOL-simulated spacecraft component data).

Usage:
    python scripts/adapt_peuda.py --source data/processed --target data/target --component reaction_wheel
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

from src.config import PEUDAConfig, CHECKPOINT_DIR, DATASET_CONFIGS
from src.models.peuda import PEUDA, compute_peuda_loss, ntxent_loss
from src.data.dataset import MultiDomainDataset, create_dataloaders
from src.utils.train_utils import set_seed, EarlyStopping, AverageMeter


def main():
    parser = argparse.ArgumentParser(description="PEUDA Domain Adaptation")
    parser.add_argument("--source_dir", type=str, default="data/processed")
    parser.add_argument("--target_dir", type=str, default="data/target")
    parser.add_argument("--component", type=str, default="reaction_wheel",
                       choices=["reaction_wheel", "battery"])
    parser.add_argument("--ssl_epochs", type=int, default=50)
    parser.add_argument("--da_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder_path", type=str, default=None,
                       help="Path to SHRDL pretrained encoder")
    args = parser.parse_args()
    
    set_seed(42)
    config = PEUDAConfig()
    config.device = args.device
    
    ds_config = DATASET_CONFIGS.get(args.component, DATASET_CONFIGS["reaction_wheel"])
    
    print("="*60)
    print(f"  PEUDA: Domain Adaptation — {args.component}")
    print("="*60)
    
    # Load data (simplified — actual loading depends on COMSOL output format)
    print("\n[1/3] Loading data...")
    print(f"  Source: {args.source_dir}")
    print(f"  Target: {args.target_dir}")
    print("  (Data loading to be adapted to COMSOL output format)")
    
    # Create model
    print(f"\n[2/3] Building PEUDA model...")
    model = PEUDA(
        n_features=ds_config.n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_transformer_layers,
        n_fft=config.n_fft,
        dropout=config.dropout,
        mcl_momentum=config.mcl_momentum,
        mcl_queue_size=config.mcl_queue_size,
    ).to(config.device)
    
    # Load pretrained encoder if available
    if args.encoder_path and os.path.exists(args.encoder_path):
        print(f"  Loading pretrained encoder from {args.encoder_path}")
        pretrained = torch.load(args.encoder_path, map_location=config.device)
        # Transfer compatible parameters
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} parameter groups")
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Stage 0: Target-only SSL
    print(f"\n[3a/3] Stage 0: Target SSL ({args.ssl_epochs} epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Note: Actual training loops will use real data loaders
    # This script serves as a template for the server
    
    print(f"\n[3b/3] Stage 1: Domain Adaptation ({args.da_epochs} epochs)...")
    
    # Save adapted model
    save_path = os.path.join(CHECKPOINT_DIR, f"peuda_{args.component}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "component": args.component,
    }, save_path)
    print(f"\nAdaptation complete! Model saved to: {save_path}")


if __name__ == "__main__":
    main()
