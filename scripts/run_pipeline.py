"""
Complete Training Pipeline — runs all three stages.

Stage 1: SHRDL pretraining on multi-source data  
Stage 2: PEUDA domain adaptation for each target component
Stage 3: PCBNN uncertainty quantification

Usage:
    python scripts/run_pipeline.py --mode all --device cuda:0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from datetime import datetime
import json

from src.config import TrainingConfig, ROOT_DIR, CHECKPOINT_DIR, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Complete Training Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["pretrain", "adapt", "uncertainty", "evaluate", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--nozzle_dir", type=str, default=r"D:\数据集\航天器")
    parser.add_argument("--target_dir", type=str, default="data/target")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    args = parser.parse_args()
    
    config = TrainingConfig()
    config.device = args.device
    
    print("="*70)
    print("  SPACECRAFT COMPONENT RUL PREDICTION — Full Pipeline")
    print("  Method: SHRDL + PEUDA + PCBNN")
    print("="*70)
    print(f"  Device: {args.device}")
    print(f"  Mode: {args.mode}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # Stage 1: Pretraining
    if args.mode in ["pretrain", "all"]:
        print("\n" + "="*70)
        print("  STAGE 1: SHRDL Self-Supervised Pretraining")
        print("="*70)
        print("  Source domains: nozzle ablation, C-MAPSS, NASA/CALCE battery")
        print("  Method: ADN encoder + CIMCL contrastive loss + MoCo")
        print("  Output: Universal degradation feature encoder")
        print("  [Run: python scripts/pretrain_shrdl.py]")
    
    # Stage 2: Domain Adaptation
    if args.mode in ["adapt", "all"]:
        for component in ["reaction_wheel", "battery"]:
            print(f"\n{'='*70}")
            print(f"  STAGE 2: PEUDA Domain Adaptation — {component}")
            print(f"{'='*70}")
            print(f"  Target: {component}")
            print(f"  Method: Dual time-freq encoder + GRL adversarial + MCL")
            print(f"  [Run: python scripts/adapt_peuda.py --component {component}]")
    
    # Stage 3: Uncertainty Quantification
    if args.mode in ["uncertainty", "all"]:
        for component in ["reaction_wheel", "battery"]:
            print(f"\n{'='*70}")
            print(f"  STAGE 3: PCBNN Uncertainty Quantification — {component}")
            print(f"{'='*70}")
            print(f"  Method: Weibull likelihood + Bayesian VI + HGRR")
            print(f"  Output: RUL + confidence intervals + risk levels")
            print(f"  [Run: python scripts/train_pcbnn.py --component {component}]")
    
    # Stage 4: Evaluation
    if args.mode in ["evaluate", "all"]:
        print(f"\n{'='*70}")
        print(f"  STAGE 4: Evaluation & Comparison")
        print(f"{'='*70}")
        print(f"  Baselines: Wiener process, LSTM, CNN-LSTM")
        print(f"  Ablation: Our method without domain adaptation")
        print(f"  Metrics: RMSE, MAE, Score, R², PICP, Advance Warning")
        print(f"  [Run: python scripts/evaluate_all.py]")
    
    print(f"\n{'='*70}")
    print(f"  Pipeline ready for execution on GPU server.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
