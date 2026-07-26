"""GEO satellite battery → NASA ground-test battery cross-domain transfer.

Competition theme: 基于跨领域退化数据迁移的航天器关键组件寿命预测

Source domain: GEO satellite PINN simulation (COMSOL-derived ODE, same Li-ion
               physics, different operating conditions: orbital shadow cycling).
Target domain: NASA B0005/B0006/B0018 ground-test cells (real-world CC/CV
               cycling, identical cell chemistry).

Transfer approach:
  1. Pre-train domain-invariant degradation encoder on GEO source (30+ traj).
  2. Fine-tune target-side projection head on NASA cells (full / few-shot).
  3. Evaluate RUL RMSE under strict14 nested LOO protocol on NASA test cells.

The source and target share the same degradation mechanism (SOH ↓, rint ↑)
but differ in: operating schedule, thermal environment, measurement channels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.geo_battery_pinn import (
    generate_multi_trajectory,
    make_split_assignment,
    simulate_geo_battery,
)
from src.data.battery_strict import (
    BATTERY_NAMES,
    FEATURE_NAMES,
    BatterySeries,
    BatteryWindowSet,
    fit_scaler,
    load_battery_mat,
    make_windows,
    materialize_battery,
)
from src.models.battery_strict import build_battery_model, RUL_SCALE, count_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEO_FEATURE_NAMES = (
    "soh", "capacity_mAh", "rint", "Tcell",
    "shadow_min", "F_DOD", "stress", "formation", "alpha_T",
)
NASA_FEATURE_NAMES = FEATURE_NAMES
SEEDS = (42, 123, 456, 2026, 3407)
TRANSFER_EPOCHS = 80
FINETUNE_EPOCHS = 40
SEQ_LEN = 16


# ---------------------------------------------------------------------------
# Shared encoder (domain-agnostic temporal backbone)
# ---------------------------------------------------------------------------


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=(kernel - 1) * dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.size(-1)
        h = self.conv(x)[..., :l]
        h = self.drop(F.gelu(self.mix(h)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class SharedDegradationEncoder(nn.Module):
    """Domain-agnostic multi-scale degradation encoder.

    Takes a projected (domain-normalised) sequence and produces a fixed-size
    degradation embedding.  Pre-trained on GEO source, then fine-tuned on
    NASA target via a domain-specific input projection.
    """

    def __init__(self, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        branch = max(16, latent_dim // 4)
        self.branch = branch
        # Input projection (set per domain)
        self.in_proj: nn.Module | None = None
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalBlock(branch, k, 1, dropout),
                CausalBlock(branch, k, 2, dropout),
            )
            for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(branch * 3)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).transpose(1, 2)
        outs = [block(bx) for bx, block in zip(torch.split(h, self.branch, 1), self.branches)]
        return self.norm(torch.cat(outs, 1).transpose(1, 2)).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class TransferRULModel(nn.Module):
    """Full transfer model: domain-specific in-proj + shared encoder + RUL head."""

    def __init__(
        self,
        n_input_features: int,
        latent_dim: int = 64,
        dropout: float = 0.1,
        rul_scale: float = float(RUL_SCALE),
    ):
        super().__init__()
        self.rul_scale = rul_scale
        branch = max(16, latent_dim // 4)
        self.in_proj = nn.Linear(n_input_features, branch * 3)
        self.encoder = SharedDegradationEncoder(latent_dim, dropout)
        self.encoder.in_proj = self.in_proj
        self.rul_head = nn.Sequential(
            nn.Linear(branch * 3, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.rul_scale * F.softplus(self.rul_head(z).squeeze(-1))


# ---------------------------------------------------------------------------
# GEO source-domain data handling
# ---------------------------------------------------------------------------


@dataclass
class GEOWindowSet:
    X: np.ndarray
    rul: np.ndarray
    trajectory_ids: np.ndarray
    split: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


def make_geo_windows(
    trajectories: list,
    *,
    seq_len: int = SEQ_LEN,
    train_only_scaler: bool = True,
) -> tuple[GEOWindowSet, np.ndarray, np.ndarray]:
    xs, ys, ids, splits = [], [], [], []
    all_feat = []
    for params, records, split in trajectories:
        feat_mat = np.column_stack([records[name] for name in GEO_FEATURE_NAMES]).astype(np.float64)
        all_feat.append((feat_mat, records["RUL"].astype(np.float64), params.trajectory_id, split))
    train_feat = np.concatenate([f for f, _, _, sp in all_feat if sp == "train"], axis=0)
    mean = train_feat.mean(0)
    scale = train_feat.std(0)
    scale[scale < 1e-10] = 1.0
    for feat_mat, rul, tid, sp in all_feat:
        norm = ((feat_mat - mean) / scale).astype(np.float32)
        for end in range(seq_len - 1, len(norm)):
            xs.append(norm[end - seq_len + 1 : end + 1])
            ys.append(rul[end])
            ids.append(tid)
            splits.append(sp)
    if not xs:
        raise ValueError("No GEO windows generated")
    return (
        GEOWindowSet(
            X=np.stack(xs).astype(np.float32),
            rul=np.asarray(ys, dtype=np.float32),
            trajectory_ids=np.asarray(ids),
            split=np.asarray(splits),
        ),
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse_val(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae_val(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def bias_val(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(b - a))


def _json_safe(v):
    if isinstance(v, np.ndarray): return _json_safe(v.tolist())
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, Mapping): return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_json_safe(x) for x in v]
    return v


def write_json(path: Path, val: object) -> None:
    path.write_text(json.dumps(_json_safe(val), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train_source(
    model: TransferRULModel,
    geo_windows: GEOWindowSet,
    device: str,
    seed: int,
    *,
    epochs: int,
    batch_size: int = 1024,
    patience: int = 20,
    rul_scale: float = float(RUL_SCALE),
) -> float:
    seed_everything(seed)
    model = model.to(device)
    x_tr = torch.as_tensor(geo_windows.X[geo_windows.split == "train"], dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(geo_windows.rul[geo_windows.split == "train"] / rul_scale, dtype=torch.float32, device=device)
    val_mask = geo_windows.split == "validation"
    x_va = torch.as_tensor(geo_windows.X[val_mask], dtype=torch.float32, device=device) if val_mask.any() else None
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * math.ceil(len(x_tr) / batch_size)))
    rng = np.random.default_rng(seed)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x_tr))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_tr[idx])
            loss = F.smooth_l1_loss(pred / rul_scale, y_tr[idx], beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if x_va is not None:
            model.eval()
            with torch.no_grad():
                vp = model(x_va).cpu().numpy()
            score = rmse_val(geo_windows.rul[val_mask], vp)
            if score < best - 1e-5:
                best, stale = score, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience: break
    if best_state: model.load_state_dict(best_state)
    return best


def finetune_nasa(
    source_model: TransferRULModel,
    train_windows: BatteryWindowSet,
    val_windows: BatteryWindowSet | None,
    device: str,
    seed: int,
    *,
    epochs: int,
    n_input_nasa: int,
    freeze_encoder: bool = False,
    batch_size: int = 512,
    patience: int = 20,
) -> tuple[TransferRULModel, int, float]:
    """Create a NASA target-side model by transferring the shared encoder weights."""
    nasa_model = TransferRULModel(n_input_nasa, latent_dim=64, rul_scale=source_model.rul_scale)
    # Transfer shared encoder and RUL head weights
    encoder_state = {k: v for k, v in source_model.encoder.state_dict().items() if "in_proj" not in k}
    nasa_model.encoder.load_state_dict(encoder_state, strict=False)
    nasa_model.rul_head.load_state_dict(source_model.rul_head.state_dict())
    if freeze_encoder:
        for name, param in nasa_model.encoder.named_parameters():
            if "in_proj" not in name:
                param.requires_grad = False
    nasa_model = nasa_model.to(device)
    seed_everything(seed)
    x_tr = torch.as_tensor(train_windows.X, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(train_windows.rul / nasa_model.rul_scale, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, nasa_model.parameters()), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * math.ceil(len(x_tr) / batch_size)))
    rng = np.random.default_rng(seed)
    val_x = torch.as_tensor(val_windows.X, dtype=torch.float32, device=device) if val_windows else None
    best, best_state, stale, best_epoch = float("inf"), None, 0, epochs
    for epoch in range(1, epochs + 1):
        nasa_model.train()
        order = rng.permutation(len(x_tr))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = nasa_model(x_tr[idx])
            mask = torch.as_tensor(train_windows.target_mask[idx], dtype=torch.bool, device=device)
            if mask.any():
                loss = F.smooth_l1_loss(pred[mask] / nasa_model.rul_scale, y_tr[idx][mask], beta=0.05)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(nasa_model.parameters(), 1.0)
                opt.step(); sched.step()
        if val_x is not None:
            nasa_model.eval()
            with torch.no_grad():
                vp = nasa_model(val_x).cpu().numpy()
            truth = val_windows.rul[val_windows.target_mask]
            if len(truth):
                score = rmse_val(truth, vp[val_windows.target_mask])
                if score < best - 1e-5:
                    best, best_epoch, stale = score, epoch, 0
                    best_state = {k: v.cpu().clone() for k, v in nasa_model.state_dict().items()}
                else:
                    stale += 1
                if stale >= 20: break
    if best_state: nasa_model.load_state_dict(best_state)
    return nasa_model, best_epoch, float(best)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_transfer_experiment(
    data_dir: Path,
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
    n_geo: int = 30,
    transfer_epochs: int = TRANSFER_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    quick: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        seeds = seeds[:1]
        transfer_epochs = min(transfer_epochs, 8)
        finetune_epochs = min(finetune_epochs, 8)
        n_geo = 8

    # 1. Generate GEO source trajectories
    print(f"Generating {n_geo} GEO source trajectories ...", flush=True)
    geo_traj = generate_multi_trajectory(n_trajectories=n_geo, seed=42)
    split_assign = make_split_assignment(geo_traj, val_fraction=0.15, test_fraction=0.15)
    geo_with_split = [(p, r, split_assign[p.trajectory_id]) for p, r in geo_traj]

    geo_windows, geo_mean, geo_scale = make_geo_windows(geo_with_split, seq_len=SEQ_LEN)
    print(f"  source windows: {len(geo_windows)} "
          f"(train={int((geo_windows.split=='train').sum())}, "
          f"val={int((geo_windows.split=='validation').sum())})", flush=True)

    # 2. Load NASA target cells
    mat_dir = data_dir / "nasa_battery" / "5. Battery Data Set"
    raws = {name: load_battery_mat(mat_dir / f"{name}.mat") for name in BATTERY_NAMES}
    series = [materialize_battery(raws[name], "strict14") for name in BATTERY_NAMES]

    # 3. Pre-train source model (per seed)
    n_geo_feat = len(GEO_FEATURE_NAMES)
    source_models = []
    source_val_rmse = []
    for seed in seeds:
        print(f"  pre-train seed={seed} ...", flush=True)
        model = TransferRULModel(n_geo_feat, latent_dim=64)
        val_rmse = train_source(model, geo_windows, device, seed, epochs=transfer_epochs)
        source_models.append(model.cpu())
        source_val_rmse.append(val_rmse)
        print(f"    source val RMSE: {val_rmse:.3f}", flush=True)

    # 4. Nested LOO fine-tuning and evaluation on NASA cells
    n_nasa_feat = len(NASA_FEATURE_NAMES)
    folds = []
    prediction_rows = []
    for holdout in [s.name for s in series]:
        outer_train = [s for s in series if s.name != holdout]
        test_series = next(s for s in series if s.name == holdout)
        scaler = fit_scaler(outer_train, NASA_FEATURE_NAMES)
        train_windows, _ = make_windows(outer_train, SEQ_LEN, scaler=scaler,
                                        feature_names=NASA_FEATURE_NAMES, min_endpoint=23)
        test_windows, _ = make_windows([test_series], SEQ_LEN, scaler=scaler,
                                       feature_names=NASA_FEATURE_NAMES, min_endpoint=23)
        # scratch model (no transfer) for comparison
        # Transfer models
        transfer_preds, scratch_preds = [], []
        for i, (source_model, seed) in enumerate(zip(source_models, seeds)):
            print(f"  [LOO {holdout}] seed={seed} fine-tune ...", flush=True)
            nasa_model, ep, _ = finetune_nasa(
                source_model, train_windows, None, device, seed,
                epochs=finetune_epochs, n_input_nasa=n_nasa_feat, freeze_encoder=False,
            )
            nasa_model.eval()
            with torch.no_grad():
                x_te = torch.as_tensor(test_windows.X, dtype=torch.float32, device=device)
                pred = nasa_model(x_te).cpu().numpy()
            transfer_preds.append(pred)
            # Scratch: same architecture, no transfer
            scratch = TransferRULModel(n_nasa_feat, latent_dim=64)
            seed_everything(seed)
            # Quick scratch training using existing battery runner approach
            scratch_trained, _, _ = finetune_nasa(
                scratch, train_windows, None, device, seed,
                epochs=finetune_epochs, n_input_nasa=n_nasa_feat, freeze_encoder=False,
            )
            scratch_trained.eval()
            with torch.no_grad():
                sp = scratch_trained(x_te).cpu().numpy()
            scratch_preds.append(sp)
        transfer_ensemble = np.mean(np.stack(transfer_preds), axis=0)
        scratch_ensemble = np.mean(np.stack(scratch_preds), axis=0)
        truth = test_windows.rul
        exact = test_windows.target_mask
        def metrics(pred, name):
            if exact.any():
                return {"name": name, "rmse": rmse_val(truth[exact], pred[exact]),
                        "mae": mae_val(truth[exact], pred[exact]),
                        "bias": bias_val(truth[exact], pred[exact]),
                        "n_exact": int(exact.sum())}
            return {"name": name, "rmse": None, "mae": None, "bias": None, "n_exact": 0,
                    "status": "right_censored"}
        fold = {
            "holdout": holdout,
            "status": test_series.status,
            "metrics": [metrics(transfer_ensemble, "transfer"), metrics(scratch_ensemble, "scratch")],
        }
        folds.append(fold)
        for i in range(len(test_windows)):
            prediction_rows.append({
                "holdout": holdout,
                "endpoint": int(test_windows.endpoints[i]),
                "rul": float(truth[i]) if np.isfinite(truth[i]) else None,
                "transfer_pred": float(transfer_ensemble[i]),
                "scratch_pred": float(scratch_ensemble[i]),
                "exact": bool(exact[i]),
            })
        tr_rmse = fold["metrics"][0]["rmse"]
        sc_rmse = fold["metrics"][1]["rmse"]
        if tr_rmse and sc_rmse:
            delta = sc_rmse - tr_rmse
            print(f"  [LOO {holdout}] transfer={tr_rmse:.3f} scratch={sc_rmse:.3f} Δ={delta:+.3f}", flush=True)

    event_folds = [f for f in folds if f["status"] == "event_observed"]
    def macro_rmse(method): 
        vals = [next(m["rmse"] for m in f["metrics"] if m["name"] == method) for f in event_folds]
        valid = [v for v in vals if v is not None]
        return float(np.mean(valid)) if valid else None

    result = {
        "schema": "geo_to_nasa_battery_transfer_v1",
        "source_domain": "GEO satellite battery PINN simulation",
        "target_domain": "NASA B0005/B0006/B0007/B0018 strict14",
        "n_geo_trajectories": n_geo,
        "transfer_epochs": transfer_epochs,
        "finetune_epochs": finetune_epochs,
        "seed_ids": list(seeds),
        "source_val_rmse": [float(v) for v in source_val_rmse],
        "folds": folds,
        "exact_event_cells": [f["holdout"] for f in event_folds],
        "macro_rmse": {"transfer": macro_rmse("transfer"), "scratch": macro_rmse("scratch")},
    }
    if result["macro_rmse"]["transfer"] and result["macro_rmse"]["scratch"]:
        result["transfer_improvement"] = result["macro_rmse"]["scratch"] - result["macro_rmse"]["transfer"]
    write_json(out_dir / "GEO_TO_NASA_TRANSFER_REPORT.json", result)
    with (out_dir / "transfer_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]) if prediction_rows else [])
        writer.writeheader(); writer.writerows(prediction_rows)
    print(json.dumps(result["macro_rmse"], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="outputs/geo_to_nasa_transfer_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--n-geo", type=int, default=30)
    parser.add_argument("--transfer-epochs", type=int, default=TRANSFER_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    t0 = time.time()
    result = run_transfer_experiment(
        Path(args.data_root), Path(args.output), device, seeds,
        n_geo=args.n_geo, transfer_epochs=args.transfer_epochs,
        finetune_epochs=args.finetune_epochs, quick=args.quick,
    )
    write_json(Path(args.output) / "run_metadata.json", {"elapsed_sec": time.time() - t0, "device": device})


if __name__ == "__main__":
    main()
