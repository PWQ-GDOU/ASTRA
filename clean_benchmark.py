"""
Leakage-free C-MAPSS benchmark and model screening.

The official test RUL labels are used only after training and checkpoint selection.
Validation is an engine-level split from the official training trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


COMMON_SENSOR_1BASED = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def seed_everything(seed: int) -> None:
    """Fix stochastic sources while allowing optimized CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def score_phm08(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA/PHM08 score; overestimation is the more severe error."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    d = yp - yt
    terms = np.where(d >= 0.0, np.exp(d / 10.0) - 1.0,
                     np.exp(-d / 13.0) - 1.0)
    return float(np.sum(terms))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


@dataclass
class CMapssData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_units: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_features: int
    train_units: np.ndarray
    val_units: np.ndarray
    rul_cap: float
    condition_centers: np.ndarray | None = None
    condition_mean: np.ndarray | None = None
    condition_std: np.ndarray | None = None


def _feature_indices(sensor_mode: str) -> np.ndarray:
    if sensor_mode in {"all24", "condnorm24"}:
        return np.arange(24, dtype=np.int64)
    if sensor_mode in {"settings14", "condnorm17"}:
        # First three columns are operating settings; remaining indices are
        # the conventional 14 non-constant sensors.
        return np.array([0, 1, 2] + [3 + s - 1 for s in COMMON_SENSOR_1BASED],
                        dtype=np.int64)
    if sensor_mode == "tts14":
        # TTSNet-style input: the 14 commonly retained degradation sensors,
        # without operating settings.
        # Raw C-MAPSS columns are unit, cycle, 3 settings, then 21 sensors.
        return np.array([3 + s - 1 for s in COMMON_SENSOR_1BASED],
                        dtype=np.int64)
    raise ValueError(f"Unknown sensor_mode={sensor_mode}")


def _exp_smooth(trajectory: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    smoothed = trajectory.astype(np.float32, copy=True)
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * trajectory[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def _fit_condition_normalizer(raw: np.ndarray, train_units: np.ndarray,
                              indices: np.ndarray, split_seed: int):
    """Fit condition clusters and selected-feature stats on training units."""
    from sklearn.cluster import KMeans

    rows = raw[np.isin(raw[:, 0], train_units)]
    settings = rows[:, 2:5].astype(np.float32)
    sensors = rows[:, 2:][:, indices].astype(np.float32)
    clusterer = KMeans(n_clusters=6, n_init=10, random_state=split_seed)
    labels = clusterer.fit_predict(settings)
    cluster_mean = np.zeros((6, sensors.shape[1]), dtype=np.float32)
    cluster_std = np.zeros_like(cluster_mean)
    global_mean = sensors.mean(axis=0).astype(np.float32)
    global_std = (sensors.std(axis=0) + 1e-8).astype(np.float32)
    for cluster_id in range(6):
        selected = sensors[labels == cluster_id]
        if len(selected) < 2:
            cluster_mean[cluster_id] = global_mean
            cluster_std[cluster_id] = global_std
        else:
            cluster_mean[cluster_id] = selected.mean(axis=0)
            cluster_std[cluster_id] = selected.std(axis=0) + 1e-8
    return clusterer, cluster_mean, cluster_std, global_mean, global_std


def _normalize_trajectory(trajectory: np.ndarray, mean: np.ndarray,
                          std: np.ndarray, condition_normalizer=None,
                          smooth: bool = False):
    if smooth:
        trajectory = _exp_smooth(trajectory)
    if condition_normalizer is None:
        return (trajectory - mean) / std
    clusterer, cluster_mean, cluster_std, _, _ = condition_normalizer
    labels = clusterer.predict(trajectory[:, :3]).astype(np.int64)
    normalized = (trajectory - mean) / std
    for cluster_id in range(cluster_mean.shape[0]):
        mask = labels == cluster_id
        if np.any(mask):
            # The first three selected columns are settings; condition
            # statistics apply to the remaining degradation channels.
            normalized[mask, 3:] = (
                trajectory[mask, 3:] - cluster_mean[cluster_id, 3:]
            ) / cluster_std[cluster_id, 3:]
    return normalized


def _read_fd(data_root: str, fd: str):
    base = Path(data_root) / "cmapss" / "6. Turbofan Engine Degradation Simulation Data Set"
    tr = pd.read_csv(base / f"train_{fd}.txt", sep=r"\s+", header=None).values
    te = pd.read_csv(base / f"test_{fd}.txt", sep=r"\s+", header=None).values
    rul = pd.read_csv(base / f"RUL_{fd}.txt", sep=r"\s+", header=None).values.reshape(-1)
    return tr, te, rul


def _split_units(units: np.ndarray, val_ratio: float, split_seed: int):
    units = np.asarray(units, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    shuffled = units.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(units) * val_ratio)))
    val_units = np.sort(shuffled[:n_val])
    train_units = np.sort(shuffled[n_val:])
    return train_units, val_units


def _make_windows(raw: np.ndarray, units: np.ndarray, indices: np.ndarray,
                  mean: np.ndarray, std: np.ndarray, seq_len: int,
                  rul_cap: float, condition_normalizer=None,
                  smooth: bool = False):
    xs, ys = [], []
    for unit in units:
        trajectory = raw[raw[:, 0] == unit, 2:][:, indices].astype(np.float32)
        n = len(trajectory)
        if n < seq_len:
            continue
        trajectory = _normalize_trajectory(trajectory, mean, std,
                                           condition_normalizer, smooth)
        for start in range(n - seq_len + 1):
            end = start + seq_len - 1
            xs.append(trajectory[start:start + seq_len])
            ys.append(min(max(n - end - 1, 0), rul_cap))
    if not xs:
        raise RuntimeError("No windows generated")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def load_cmapss(data_root: str, fd: str, seq_len: int = 30,
                rul_cap: float = 125.0, val_ratio: float = 0.2,
                split_seed: int = 2026, sensor_mode: str = "all24") -> CMapssData:
    tr, te, official_rul = _read_fd(data_root, fd)
    indices = _feature_indices(sensor_mode)
    units = np.unique(tr[:, 0]).astype(np.int64)
    train_units, val_units = _split_units(units, val_ratio, split_seed)

    smooth = sensor_mode == "tts14"
    if smooth:
        train_rows = np.concatenate([
            _exp_smooth(
                tr[tr[:, 0] == unit, 2:][:, indices].astype(np.float32)
            )
            for unit in train_units
        ], axis=0)
        mean = train_rows.min(axis=0, keepdims=True)
        std = train_rows.max(axis=0, keepdims=True) - mean + 1e-8
    else:
        train_rows = tr[np.isin(tr[:, 0], train_units), 2:][:, indices].astype(np.float32)
        mean = train_rows.mean(axis=0, keepdims=True)
        std = train_rows.std(axis=0, keepdims=True) + 1e-8

    condition_normalizer = None
    if sensor_mode in {"condnorm24", "condnorm17"}:
        condition_normalizer = _fit_condition_normalizer(
            tr, train_units, indices, split_seed
        )

    X_train, y_train = _make_windows(
        tr, train_units, indices, mean, std, seq_len, rul_cap,
        condition_normalizer, smooth
    )
    X_val, y_val = _make_windows(
        tr, val_units, indices, mean, std, seq_len, rul_cap,
        condition_normalizer, smooth
    )

    test_x, test_y, test_units = [], [], []
    for unit in np.unique(te[:, 0]).astype(np.int64):
        trajectory = te[te[:, 0] == unit, 2:][:, indices].astype(np.float32)
        trajectory = _normalize_trajectory(trajectory, mean, std,
                                           condition_normalizer, smooth)
        # Keep every official test engine. For short trajectories, repeat the
        # first observed normalized state on the left; labels remain Unit-ID
        # aligned with RUL_FD00*.txt.
        if len(trajectory) < seq_len:
            pad = np.repeat(trajectory[:1], seq_len - len(trajectory), axis=0)
            trajectory = np.concatenate([pad, trajectory], axis=0)
        test_x.append(trajectory[-seq_len:])
        # The official RUL file is ordered by engine id, not retained index.
        test_y.append(float(official_rul[int(unit) - 1]))
        test_units.append(unit)

    return CMapssData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=np.stack(test_x).astype(np.float32),
        y_test=np.asarray(test_y, dtype=np.float32),
        test_units=np.asarray(test_units, dtype=np.int64),
        mean=mean, std=std, n_features=len(indices),
        train_units=train_units, val_units=val_units, rul_cap=rul_cap,
        condition_centers=(None if condition_normalizer is None else
                           condition_normalizer[0].cluster_centers_),
        condition_mean=(None if condition_normalizer is None else
                        condition_normalizer[1]),
        condition_std=(None if condition_normalizer is None else
                       condition_normalizer[2]),
    )


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, padding=padding)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        h = F.gelu(self.conv(x))
        h = h[..., :length]
        h = self.dropout(self.mix(h))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class V2TCN(nn.Module):
    def __init__(self, n_features: int, d_model: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [TCNBlock(d_model, 3, dropout=dropout) for _ in range(4)] +
            [TCNBlock(d_model, 5, dropout=dropout) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 6, dropout=dropout,
                                           batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.transpose(1, 2))
        a, _ = self.attn(h, h, h, need_weights=False)
        return (h + a).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class MultiScaleTCN(nn.Module):
    """Parallel local/mid/long temporal branches with shared output width."""
    def __init__(self, n_features: int, d_model: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        branch = d_model // 3
        self.proj = nn.Linear(n_features, d_model)
        self.branches = nn.ModuleList([
            nn.ModuleList([TCNBlock(branch, 3, dropout=dropout) for _ in range(3)]),
            nn.ModuleList([TCNBlock(branch, 5, dropout=dropout) for _ in range(3)]),
            nn.ModuleList([TCNBlock(branch, 7, dropout=dropout) for _ in range(3)]),
        ])
        self.norm = nn.LayerNorm(branch * 3)
        self.attn = nn.MultiheadAttention(branch * 3, 6, dropout=dropout,
                                           batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(branch * 3, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).transpose(1, 2)
        chunks = torch.chunk(h, 3, dim=1)
        outputs = []
        for chunk, branch in zip(chunks, self.branches):
            for block in branch:
                chunk = block(chunk)
            outputs.append(chunk)
        h = self.norm(torch.cat(outputs, dim=1).transpose(1, 2))
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.head((h + a).mean(dim=1))


class TTSNetInspired(nn.Module):
    """Paper-inspired TCN/Transformer/sensor-fusion encoder.

    This is a compact implementation of the architectural idea, not a claim
    of exact reproduction of the published TTSNet code.
    """
    def __init__(self, n_features: int, d_model: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        if n_features != 14:
            raise ValueError("TTSNetInspired requires tts14 input")
        branch = d_model // 3
        self.tcn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_features, branch, kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Conv1d(branch, branch, 3, padding=1),
                nn.GELU(),
            ) for kernel in (3, 5, 7)
        ])
        self.tcn_norm = nn.LayerNorm(d_model)
        self.transformer_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=6, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.sensor_score = nn.Linear(n_features, n_features)
        self.sensor_proj = nn.Linear(n_features, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tcn = torch.cat([
            branch(x.transpose(1, 2)).transpose(1, 2)
            for branch in self.tcn_branches
        ], dim=-1)
        tcn = self.tcn_norm(tcn).mean(dim=1)
        transformer = self.transformer(self.transformer_proj(x)).mean(dim=1)
        weights = torch.softmax(self.sensor_score(x), dim=-1)
        sensor = self.sensor_proj(x * weights).mean(dim=1)
        fused = self.fuse(torch.cat([tcn, transformer, sensor], dim=-1))
        return self.head(fused)


class ConditionSplitTCN(nn.Module):
    """Separate operating-condition and sensor streams with gated fusion."""
    def __init__(self, n_features: int, d_model: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        if n_features != 24:
            raise ValueError("ConditionSplitTCN requires all24 input features")
        self.sensor_proj = nn.Linear(21, d_model)
        self.condition_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.sensor_blocks = nn.ModuleList(
            [TCNBlock(d_model, 3, dropout=dropout) for _ in range(4)] +
            [TCNBlock(d_model, 5, dropout=dropout) for _ in range(3)]
        )
        self.condition_attn = nn.MultiheadAttention(d_model, 6,
                                                     dropout=dropout,
                                                     batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model),
                                  nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = self.condition_proj(x[..., :3])
        h = self.sensor_proj(x[..., 3:]).transpose(1, 2)
        for block in self.sensor_blocks:
            h = block(h)
        h = h.transpose(1, 2)
        condition_context, _ = self.condition_attn(h, cond, cond,
                                                   need_weights=False)
        gate = self.gate(torch.cat([h, condition_context], dim=-1))
        h = self.norm(h + gate * condition_context)
        return self.head(h.mean(dim=1))


def make_model(name: str, n_features: int) -> nn.Module:
    if name == "ttsnet":
        return TTSNetInspired(n_features)
    if name == "v2":
        return V2TCN(n_features)
    if name == "multiscale":
        return MultiScaleTCN(n_features)
    if name == "condition":
        return ConditionSplitTCN(n_features)
    raise ValueError(f"Unknown model={name}")


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def _predict_batched(model: nn.Module, x: torch.Tensor,
                     batch_size: int) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            with torch.autocast(device_type=x.device.type,
                                dtype=torch.float16,
                                enabled=x.is_cuda):
                pred = model(x[start:start + batch_size]).squeeze(-1)
            predictions.append(pred.float().cpu())
    return torch.cat(predictions).numpy()


def train_one(model: nn.Module, data: CMapssData, seed: int, device: str,
              epochs: int, batch_size: int, lr: float, over_weight: float,
              patience: int):
    seed_everything(seed)
    model.to(device)
    Xtr = torch.as_tensor(data.X_train, device=device)
    ytr = torch.as_tensor(data.y_train, device=device)
    Xva = torch.as_tensor(data.X_val, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    # FD002 labels yield large early MSE values. A modest initial scale avoids
    # overflow-induced skipped optimizer steps on Turing-generation FP16 GPUs.
    scaler = torch.amp.GradScaler("cuda", enabled=Xtr.is_cuda,
                                  init_scale=256.0, growth_interval=512)
    steps = epochs * math.ceil(len(data.X_train) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bi in _batch_indices(len(data.X_train), batch_size, rng):
            target = ytr[bi]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=Xtr.device.type,
                                dtype=torch.float16,
                                enabled=Xtr.is_cuda):
                pred = model(Xtr[bi]).squeeze(-1)
                # Keep the regression loss in FP32 even when feature encoding
                # uses tensor cores, preserving stable large-error gradients.
                pred32 = pred.float()
                diff = pred32 - target
                loss = (F.mse_loss(pred32, target) +
                        over_weight * F.relu(diff).mean())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))

        model.eval()
        val_pred = _predict_batched(model, Xva, batch_size * 4)
        val_rmse = rmse(data.y_val, val_pred)
        val_score = score_phm08(data.y_val, val_pred)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_rmse": val_rmse, "val_score": val_score})
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"E{epoch:03d} loss={np.mean(losses):.4f} "
                  f"val_rmse={val_rmse:.3f} best={best_val:.3f}", flush=True)
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint captured")
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.as_tensor(data.X_test, device=device)
    test_pred = _predict_batched(model, test_x, batch_size * 4)
    elapsed = time.time() - start_time
    capped_test_y = np.minimum(data.y_test, data.rul_cap)
    result = {
        "seed": seed, "best_epoch": best_epoch,
        "val_rmse": best_val,
        # Raw official RUL is the primary metric. The capped variant is also
        # reported because many C-MAPSS papers cap both train and test labels.
        "test_rmse": rmse(data.y_test, test_pred),
        "test_score": score_phm08(data.y_test, test_pred),
        "test_rmse_cap": rmse(capped_test_y, test_pred),
        "test_score_cap": score_phm08(capped_test_y, test_pred),
        "n_params": sum(p.numel() for p in model.parameters()),
        "seconds": elapsed,
        "test_units": data.test_units.tolist(),
        "predictions": test_pred.tolist(),
        "history": history,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--fd", default="FD002")
    parser.add_argument("--model",
                        choices=["v2", "multiscale", "condition", "ttsnet"],
                        default="v2")
    parser.add_argument("--sensor-mode",
                        choices=["all24", "settings14", "condnorm24",
                                 "condnorm17", "tts14"],
                        default="all24")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--rul-cap", type=float, default=125.0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--over-weight", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/clean_benchmark")
    args = parser.parse_args()

    data = load_cmapss(args.data_root, args.fd, args.seq_len, args.rul_cap,
                       split_seed=args.split_seed, sensor_mode=args.sensor_mode)
    print(f"FD={args.fd} model={args.model} sensors={args.sensor_mode}", flush=True)
    print(f"train={data.X_train.shape} val={data.X_val.shape} "
          f"test={data.X_test.shape} train_units={len(data.train_units)} "
          f"val_units={len(data.val_units)}", flush=True)
    os.makedirs(args.output, exist_ok=True)
    all_results = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        print(f"\n=== seed {seed} ===", flush=True)
        seed_everything(seed)
        model = make_model(args.model, data.n_features)
        result = train_one(model, data, seed, args.device, args.epochs,
                           args.batch_size, args.lr, args.over_weight,
                           args.patience)
        result.update({"fd": args.fd, "model": args.model,
                       "sensor_mode": args.sensor_mode,
                       "seq_len": args.seq_len, "rul_cap": args.rul_cap,
                       "split_seed": args.split_seed})
        all_results.append(result)
        path = Path(args.output) / f"{args.fd}_{args.model}_{args.sensor_mode}_s{seed}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        checkpoint_path = Path(args.output) / (
            f"{args.fd}_{args.model}_{args.sensor_mode}_s{seed}.pt"
        )
        torch.save({
            "model": model.state_dict(),
            "mean": data.mean,
            "std": data.std,
            "condition_centers": data.condition_centers,
            "condition_mean": data.condition_mean,
            "condition_std": data.condition_std,
            "train_units": data.train_units,
            "val_units": data.val_units,
            "config": {
                "fd": args.fd, "model": args.model,
                "sensor_mode": args.sensor_mode,
                "seq_len": args.seq_len, "rul_cap": args.rul_cap,
                "split_seed": args.split_seed, "seed": seed,
            },
        }, checkpoint_path)
        print(f"RESULT seed={seed} val_rmse={result['val_rmse']:.3f} "
              f"test_rmse={result['test_rmse']:.3f} "
              f"test_score={result['test_score']:.1f}", flush=True)
    summary = {
        "config": vars(args),
        "results": all_results,
        "mean_test_rmse": float(np.mean([r["test_rmse"] for r in all_results])),
        "std_test_rmse": float(np.std([r["test_rmse"] for r in all_results])),
        "mean_test_rmse_cap": float(np.mean(
            [r["test_rmse_cap"] for r in all_results]
        )),
        "std_test_rmse_cap": float(np.std(
            [r["test_rmse_cap"] for r in all_results]
        )),
        "mean_test_score": float(np.mean([r["test_score"] for r in all_results])),
        "mean_test_score_cap": float(np.mean(
            [r["test_score_cap"] for r in all_results]
        )),
    }
    (Path(args.output) / f"{args.fd}_{args.model}_{args.sensor_mode}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
