"""Auditable nozzle-to-target cross-component transfer protocol (v3).

Positive transfer is assessed separately for NASA strict14 batteries and the
FEMTO/PRONOSTIA reaction-wheel mechanical proxy.  This module deliberately
does not reuse the v2 feature normalization or shared cycle-position head:

* source and target have independent scalar input projections;
* only the GRU degradation encoder is transferred;
* target health calibration is fitted from the fine-tuning training prefix;
* the outer holdout never enters a target scaler, adapter fit, or validation;
* transfer, scratch, and frozen-encoder ablation have the same target budget.

The default command supervises isolated child cells.  ``--worker-json`` is an
implementation detail used by that supervisor and is also useful for reruns of
a single failed cell.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_data_module(stem: str):
    """Load a standalone strict-data module without eager package side effects.

    Some DCU project snapshots have a legacy ``src.data.__init__`` which
    imports optional, unrelated notebook dependencies.  The three strict
    loaders below are self-contained, so loading their files directly keeps
    the v3 runner tied to its declared NumPy/SciPy/PyTorch environment.
    """
    path = ROOT / "src" / "data" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_cross_transfer_v3_{stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strict data module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_battery_data = _load_data_module("battery_strict")
_femto_data = _load_data_module("femto_strict")
_nozzle_data = _load_data_module("nozzle_multitrajectory")
BATTERY_NAMES = _battery_data.BATTERY_NAMES
load_battery_mat = _battery_data.load_battery_mat
materialize_battery = _battery_data.materialize_battery
BEARING_NAMES = _femto_data.BEARING_NAMES
load_femto_zip = _femto_data.load_femto_zip
load_nozzle_trajectories = _nozzle_data.load_nozzle_trajectories


SCHEMA = "cross_component_transfer_v3"
SEEDS = (42, 123, 456, 2026, 3407)
BATTERY_EVENT_FOLDS = ("B0005", "B0006", "B0018")
BATTERY_CENSORED = "B0007"
N_SHOTS = {"battery": (1, 2), "rw_proxy": (1, 2, 3)}
N_TRIALS = 3
SOURCE_SPLIT_SEED = 20260819
TRIAL_SELECTION_SEED = 3407001
SEQ_LEN = {"battery": 16, "rw_proxy": 20, "source": 5}
EARLY_FRAC = 0.20
VAL_FRAC = 0.20
VAL_GAP = 4
HIDDEN = 64
ADAPTER_DIM = 24
DROPOUT = 0.10
DEFAULT_SOURCE_EPOCHS = 180
DEFAULT_TARGET_EPOCHS = 180
DEFAULT_PATIENCE = 24
DEFAULT_BATCH = 512


@dataclass(frozen=True)
class HealthTransform:
    """A train-only scalar calibration, with rising damage mapped to health."""

    low: float
    scale: float
    train_units: tuple[str, ...]
    train_point_count: int
    direction: str = "health = 1 - (signal - low) / scale"

    def transform(self, values: np.ndarray) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float32)
        return np.clip(1.0 - (signal - self.low) / self.scale, 0.0, 1.0).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "low": float(self.low),
            "scale": float(self.scale),
            "train_units": list(self.train_units),
            "train_point_count": int(self.train_point_count),
            "direction": self.direction,
        }


@dataclass
class WindowData:
    raw_x: np.ndarray
    y: np.ndarray
    exact: np.ndarray
    lower: np.ndarray
    endpoints: np.ndarray
    units: np.ndarray

    def __len__(self) -> int:
        return int(self.raw_x.shape[0])

    def transformed(self, transform: HealthTransform, label_scale: float) -> "PreparedData":
        return PreparedData(
            x=transform.transform(self.raw_x)[..., None],
            y=(self.y / max(label_scale, 1.0e-8)).astype(np.float32),
            exact=self.exact.astype(bool),
            lower=(self.lower / max(label_scale, 1.0e-8)).astype(np.float32),
            endpoints=self.endpoints.astype(np.int64),
            units=self.units.astype(object),
        )


@dataclass
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    exact: np.ndarray
    lower: np.ndarray
    endpoints: np.ndarray
    units: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True)
class TargetRecord:
    name: str
    raw_signal: np.ndarray
    rul: np.ndarray
    exact: np.ndarray
    lower: np.ndarray
    status: str
    metadata: Mapping[str, object]

    def __len__(self) -> int:
        return int(self.raw_signal.size)


@dataclass
class TargetBundle:
    train: PreparedData
    val: PreparedData | None
    test: PreparedData
    transform: HealthTransform
    label_scale: float
    split_audit: dict[str, object]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_safe(row.get(name)) for name in fieldnames})
    temp.replace(path)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_heartbeat(path: Path | None, **fields: Any) -> None:
    if path is None:
        return
    atomic_write_json(path, {"updated_at_utc": utc_now(), **fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (2**32 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(truth) & np.isfinite(pred)
    return float(np.sqrt(np.mean((truth[valid] - pred[valid]) ** 2))) if np.any(valid) else float("nan")


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(truth) & np.isfinite(pred)
    return float(np.mean(np.abs(truth[valid] - pred[valid]))) if np.any(valid) else float("nan")


def prediction_metrics(truth: np.ndarray, pred: np.ndarray, label_scale: float) -> dict[str, float | int | None]:
    return {
        "raw_rmse": rmse(truth, pred),
        "mae": mae(truth, pred),
        "bias": float(np.nanmean(np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64))),
        "normalized_rmse": rmse(truth, pred) / max(float(label_scale), 1.0),
        "n_points": int(np.sum(np.isfinite(truth) & np.isfinite(pred))),
    }


def direction_correlation(health: np.ndarray, rul: np.ndarray, exact: np.ndarray | None = None) -> float:
    health = np.asarray(health, dtype=np.float64).reshape(-1)
    rul = np.asarray(rul, dtype=np.float64).reshape(-1)
    valid = np.isfinite(health) & np.isfinite(rul)
    if exact is not None:
        valid &= np.asarray(exact, dtype=bool).reshape(-1)
    if np.sum(valid) < 3 or np.std(health[valid]) < 1.0e-12 or np.std(rul[valid]) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(health[valid], rul[valid])[0, 1])


def fit_health_transform(values_by_unit: Mapping[str, np.ndarray]) -> HealthTransform:
    if not values_by_unit:
        raise ValueError("Target health calibration requires at least one training unit")
    values = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in values_by_unit.values()])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Target health calibration has no finite values")
    low = float(np.min(values))
    high = float(np.max(values))
    return HealthTransform(
        low=low,
        scale=max(high - low, 1.0e-8),
        train_units=tuple(sorted(values_by_unit)),
        train_point_count=int(values.size),
    )


def assert_holdout_excluded(holdout: str, transform: HealthTransform, train: PreparedData,
                            val: PreparedData | None) -> None:
    if holdout in transform.train_units:
        raise AssertionError(f"Holdout {holdout} entered target health calibration")
    if holdout in set(train.units.astype(str)):
        raise AssertionError(f"Holdout {holdout} entered target fit data")
    if val is not None and holdout in set(val.units.astype(str)):
        raise AssertionError(f"Holdout {holdout} entered target validation data")


def select_trial_units(available: Sequence[str], n_shot: int, trial: int, *, domain: str,
                       holdout: str) -> tuple[str, ...]:
    """Stable selection independent of Python's randomized ``hash()``."""
    names = tuple(sorted(str(name) for name in available))
    if not 1 <= n_shot <= len(names):
        raise ValueError(f"n_shot={n_shot} incompatible with {len(names)} available units")
    rng = np.random.default_rng(stable_seed(TRIAL_SELECTION_SEED, domain, holdout, n_shot, trial))
    picked = rng.choice(len(names), size=n_shot, replace=False)
    return tuple(names[int(index)] for index in np.sort(picked))


def _empty_window_data(seq_len: int) -> WindowData:
    return WindowData(
        raw_x=np.empty((0, seq_len), dtype=np.float32),
        y=np.empty(0, dtype=np.float32),
        exact=np.empty(0, dtype=bool),
        lower=np.empty(0, dtype=np.float32),
        endpoints=np.empty(0, dtype=np.int64),
        units=np.empty(0, dtype=object),
    )


def _combine_window_data(parts: Sequence[WindowData], seq_len: int) -> WindowData:
    usable = [item for item in parts if len(item)]
    if not usable:
        return _empty_window_data(seq_len)
    return WindowData(
        raw_x=np.concatenate([item.raw_x for item in usable], axis=0),
        y=np.concatenate([item.y for item in usable]),
        exact=np.concatenate([item.exact for item in usable]),
        lower=np.concatenate([item.lower for item in usable]),
        endpoints=np.concatenate([item.endpoints for item in usable]),
        units=np.concatenate([item.units for item in usable]),
    )


def make_windows_for_record(record: TargetRecord, seq_len: int, *, last_endpoint: int | None = None) -> WindowData:
    final = len(record) - 1 if last_endpoint is None else min(int(last_endpoint), len(record) - 1)
    first = seq_len - 1
    if final < first:
        return _empty_window_data(seq_len)
    raw_x, y, exact, lower, endpoints = [], [], [], [], []
    for endpoint in range(first, final + 1):
        raw_x.append(record.raw_signal[endpoint - seq_len + 1 : endpoint + 1])
        y.append(record.rul[endpoint])
        exact.append(record.exact[endpoint])
        lower.append(record.lower[endpoint])
        endpoints.append(endpoint)
    return WindowData(
        raw_x=np.asarray(raw_x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        exact=np.asarray(exact, dtype=bool),
        lower=np.asarray(lower, dtype=np.float32),
        endpoints=np.asarray(endpoints, dtype=np.int64),
        units=np.asarray([record.name] * len(endpoints), dtype=object),
    )


def _prepared_from_window(window: WindowData, transform: HealthTransform, label_scale: float) -> PreparedData:
    return window.transformed(transform, label_scale)


def _prefix_train_validation(record: TargetRecord, seq_len: int, early_frac: float,
                             val_frac: float, val_gap: int) -> tuple[WindowData, WindowData]:
    early_last = max(seq_len - 1, int(math.ceil(len(record) * early_frac)) - 1)
    early = make_windows_for_record(record, seq_len, last_endpoint=early_last)
    if len(early) < 6:
        return early, _empty_window_data(seq_len)
    n_val = max(2, int(math.ceil(len(early) * val_frac)))
    # Short strict14 prefixes may hold only seven sequence windows.  Retain
    # the chronological validation tail by shrinking the optional gap first.
    effective_gap = min(max(0, val_gap), max(0, len(early) - n_val - 3))
    train_end = len(early) - n_val - effective_gap
    if train_end < 3:
        return early, _empty_window_data(seq_len)
    return (
        WindowData(
            raw_x=early.raw_x[:train_end], y=early.y[:train_end], exact=early.exact[:train_end],
            lower=early.lower[:train_end], endpoints=early.endpoints[:train_end], units=early.units[:train_end],
        ),
        WindowData(
            raw_x=early.raw_x[-n_val:], y=early.y[-n_val:], exact=early.exact[-n_val:],
            lower=early.lower[-n_val:], endpoints=early.endpoints[-n_val:], units=early.units[-n_val:],
        ),
    )


def build_target_bundle(records: Mapping[str, TargetRecord], *, selected: Sequence[str], holdout: str,
                        seq_len: int, early_frac: float = EARLY_FRAC, val_frac: float = VAL_FRAC,
                        val_gap: int = VAL_GAP) -> TargetBundle:
    if holdout in selected:
        raise ValueError("Holdout cannot be a fine-tuning unit")
    train_parts, val_parts = [], []
    calibration: dict[str, np.ndarray] = {}
    audit_units: dict[str, dict[str, object]] = {}
    for name in selected:
        record = records[name]
        if not np.any(record.exact):
            raise ValueError(f"Fine-tuning unit {name} has no exact target labels")
        train_part, val_part = _prefix_train_validation(record, seq_len, early_frac, val_frac, val_gap)
        if not len(train_part):
            raise ValueError(f"Fine-tuning unit {name} has no train windows")
        train_parts.append(train_part)
        val_parts.append(val_part)
        calibration[name] = train_part.raw_x.reshape(-1)
        audit_units[name] = {
            "train_endpoints": train_part.endpoints.tolist(),
            "validation_endpoints": val_part.endpoints.tolist(),
            "early_last_endpoint": int(max(np.r_[train_part.endpoints, val_part.endpoints])),
        }
    raw_train = _combine_window_data(train_parts, seq_len)
    raw_val = _combine_window_data(val_parts, seq_len)
    if not np.all(raw_train.exact):
        raise ValueError("Fine-tuning labels must be exact for this protocol")
    label_scale = max(float(np.nanmax(raw_train.y)), 1.0)
    transform = fit_health_transform(calibration)
    train = _prepared_from_window(raw_train, transform, label_scale)
    val = _prepared_from_window(raw_val, transform, label_scale) if len(raw_val) else None
    raw_test = make_windows_for_record(records[holdout], seq_len)
    test = _prepared_from_window(raw_test, transform, label_scale)
    assert_holdout_excluded(holdout, transform, train, val)
    return TargetBundle(
        train=train,
        val=val,
        test=test,
        transform=transform,
        label_scale=label_scale,
        split_audit={
            "holdout": holdout,
            "selected_units": list(selected),
            "unit_splits": audit_units,
            "validation_mode": "chronological_prefix_with_adaptive_gap",
            "early_frac": early_frac,
            "val_frac": val_frac,
            "val_gap_windows": val_gap,
        },
    )


class DomainInputProjection(nn.Module):
    def __init__(self, adapter_dim: int = ADAPTER_DIM) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, adapter_dim),
            nn.GELU(),
            nn.LayerNorm(adapter_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SharedDegradationEncoder(nn.Module):
    def __init__(self, adapter_dim: int = ADAPTER_DIM, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.gru = nn.GRU(adapter_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output[:, -1]


class TargetRULModel(nn.Module):
    """Target adapter + copied shared encoder + independently initialized head."""

    def __init__(self, adapter_dim: int = ADAPTER_DIM, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.projection = DomainInputProjection(adapter_dim)
        self.encoder = SharedDegradationEncoder(adapter_dim, hidden, dropout)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.rul_head(self.encoder(self.projection(x))).squeeze(-1))


def _training_loss(prediction: torch.Tensor, target: torch.Tensor, exact: torch.Tensor,
                   lower: torch.Tensor) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    if bool(exact.any()):
        terms.append(F.smooth_l1_loss(prediction[exact], target[exact], beta=0.08))
    censored = ~exact
    if bool(censored.any()):
        terms.append(0.5 * F.relu(lower[censored] - prediction[censored]).square().mean())
    if not terms:
        raise ValueError("Batch has neither exact nor censored labels")
    return sum(terms) / len(terms)


def train_model(model: TargetRULModel, train: PreparedData, val: PreparedData | None, *, device: str,
                seed: int, epochs: int, patience: int, batch_size: int, freeze_encoder: bool,
                heartbeat: Path | None = None, heartbeat_fields: Mapping[str, Any] | None = None) -> tuple[TargetRULModel, dict[str, object]]:
    seed_everything(seed)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not freeze_encoder)
    parameters = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5.0e-4, weight_decay=1.0e-4)
    batches = max(1, math.ceil(len(train) / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * batches))
    rng = np.random.default_rng(seed)
    x_train = torch.as_tensor(train.x, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train.y, dtype=torch.float32, device=device)
    m_train = torch.as_tensor(train.exact, dtype=torch.bool, device=device)
    l_train = torch.as_tensor(train.lower, dtype=torch.float32, device=device)
    val_tensors = None
    if val is not None and len(val):
        val_tensors = (
            torch.as_tensor(val.x, dtype=torch.float32, device=device),
            torch.as_tensor(val.y, dtype=torch.float32, device=device),
            torch.as_tensor(val.exact, dtype=torch.bool, device=device),
            torch.as_tensor(val.lower, dtype=torch.float32, device=device),
        )
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    stopped_epoch = epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            prediction = model(x_train[indices])
            loss = _training_loss(prediction, y_train[indices], m_train[indices], l_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                prediction = model(val_tensors[0])
                score = float(torch.sqrt(torch.mean((prediction[val_tensors[2]] - val_tensors[1][val_tensors[2]]) ** 2))) if bool(val_tensors[2].any()) else float(_training_loss(prediction, val_tensors[1], val_tensors[2], val_tensors[3]))
            if score < best_score - 1.0e-6:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                stopped_epoch = epoch
                break
        else:
            best_epoch = epoch
        if heartbeat is not None and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            heartbeat_payload = dict(heartbeat_fields or {})
            heartbeat_payload.update({"stage": "training", "epoch": epoch})
            write_heartbeat(heartbeat, **heartbeat_payload)
    if best_state is not None:
        model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, {
        "freeze_encoder": bool(freeze_encoder),
        "epochs_budget": int(epochs),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "best_validation_rmse_norm": best_score if np.isfinite(best_score) else None,
        "n_train": int(len(train)),
        "n_validation": int(len(val) if val is not None else 0),
    }


@torch.no_grad()
def predict_model(model: TargetRULModel, data: PreparedData, device: str, label_scale: float) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    output = model(x).detach().cpu().numpy() * label_scale
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Model produced non-finite target predictions")
    return output.astype(np.float32)


def ridge_features(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)[:, :, 0]
    time_axis = np.arange(values.shape[1], dtype=np.float64)
    centered = time_axis - time_axis.mean()
    denom = max(float(np.square(centered).sum()), 1.0e-8)
    slope = (values * centered).sum(axis=1) / denom
    return np.column_stack([values.mean(axis=1), values[:, -1], values.std(axis=1), slope])


def fit_ridge(train: PreparedData, alpha: float = 1.0) -> np.ndarray:
    features = ridge_features(train.x)
    x = np.column_stack([np.ones(len(features)), features])
    y = train.y.astype(np.float64)
    identity = np.eye(x.shape[1])
    identity[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * identity, x.T @ y)


def predict_ridge(beta: np.ndarray, target: PreparedData, label_scale: float) -> np.ndarray:
    features = ridge_features(target.x)
    x = np.column_stack([np.ones(len(features)), features])
    return np.maximum(x @ beta, 0.0).astype(np.float32) * label_scale


def fit_time_trend(train: PreparedData, label_scale: float) -> np.ndarray:
    x = train.endpoints.astype(np.float64)
    matrix = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(matrix, train.y.astype(np.float64), rcond=None)[0]


def predict_time_trend(beta: np.ndarray, target: PreparedData, label_scale: float) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(target)), target.endpoints.astype(np.float64)])
    return np.maximum(matrix @ beta, 0.0).astype(np.float32) * label_scale


def _source_health(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    scale = max(float(np.max(depth) - np.min(depth)), 1.0e-8)
    return (1.0 - (depth - np.min(depth)) / scale).astype(np.float32)


def _source_windows(trajectories: Sequence[Any], seq_len: int) -> PreparedData:
    x_values, targets, masks, lowers, endpoints, units = [], [], [], [], [], []
    for trajectory in trajectories:
        health = _source_health(trajectory.depth_mm)
        first = seq_len - 1
        if len(trajectory) <= first:
            continue
        exact = trajectory.target_mask.astype(bool)
        if np.any(exact):
            local_scale = max(float(trajectory.rul_s[first]), 1.0) if bool(exact[first]) else max(float(np.nanmax(trajectory.rul_s[exact])), 1.0)
        else:
            local_scale = max(float(trajectory.lower_bound_s[first]), 1.0)
        for endpoint in range(first, len(trajectory)):
            x_values.append(health[endpoint - seq_len + 1 : endpoint + 1])
            targets.append(float(trajectory.rul_s[endpoint] / local_scale) if exact[endpoint] else 0.0)
            masks.append(bool(exact[endpoint]))
            lowers.append(float(trajectory.lower_bound_s[endpoint] / local_scale))
            endpoints.append(endpoint)
            units.append(trajectory.manifest.trajectory_id)
    if not x_values:
        raise ValueError("No source windows created")
    return PreparedData(
        x=np.asarray(x_values, dtype=np.float32)[..., None],
        y=np.asarray(targets, dtype=np.float32),
        exact=np.asarray(masks, dtype=bool),
        lower=np.asarray(lowers, dtype=np.float32),
        endpoints=np.asarray(endpoints, dtype=np.int64),
        units=np.asarray(units, dtype=object),
    )


def load_source_split(nozzle_path: Path, *, source_limit: int | None = None) -> tuple[dict[str, PreparedData], dict[str, object]]:
    trajectories = load_nozzle_trajectories(nozzle_path, feature_tier="estimated")
    ids = tuple(sorted(trajectories))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate nozzle trajectory identifiers")
    if len(ids) != 200 and source_limit is None:
        raise ValueError(f"v3 source protocol requires exactly 200 nozzle trajectories, found {len(ids)}")
    rng = np.random.default_rng(SOURCE_SPLIT_SEED)
    chosen = [ids[int(index)] for index in rng.permutation(len(ids))]
    if source_limit is not None:
        chosen = chosen[: max(2, min(int(source_limit), len(chosen)))]
    n_total = len(chosen)
    n_test = 0 if n_total < 8 else max(1, int(round(n_total * 0.15)))
    n_val = 1 if n_total < 8 else max(1, int(round(n_total * 0.15)))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError("Source split has no training trajectory")
    partitions = {
        "train": chosen[:n_train],
        "validation": chosen[n_train : n_train + n_val],
        "source_test": chosen[n_train + n_val :],
    }
    result = {name: _source_windows([trajectories[item] for item in values], SEQ_LEN["source"]) for name, values in partitions.items() if values}
    direction_values = []
    for name in chosen:
        item = trajectories[name]
        direction_values.append(direction_correlation(_source_health(item.depth_mm), item.rul_s, item.target_mask))
    direction = float(np.nanmean(np.asarray(direction_values, dtype=np.float64)))
    if not np.isfinite(direction) or direction <= 0.0:
        raise ValueError(f"Nozzle health/RUL direction check failed: correlation={direction}")
    return result, {
        "file": str(nozzle_path),
        "sha256": sha256_file(nozzle_path),
        "trajectory_count": len(ids),
        "selected_trajectory_count": n_total,
        "split_seed": SOURCE_SPLIT_SEED,
        "splits": partitions,
        "event_observed_count": int(sum(item.manifest.status == "event_observed" for item in trajectories.values())),
        "right_censored_count": int(sum(item.manifest.status == "right_censored" for item in trajectories.values())),
        "mean_health_rul_correlation": direction,
    }


def load_battery_records(battery_dir: Path) -> tuple[dict[str, TargetRecord], dict[str, object]]:
    records: dict[str, TargetRecord] = {}
    manifest: dict[str, object] = {"path": str(battery_dir), "files": {}}
    for name in BATTERY_NAMES:
        path = battery_dir / f"{name}.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        series = materialize_battery(load_battery_mat(path), "strict14")
        records[name] = TargetRecord(
            name=name,
            # The shared target transform expects a damage signal which rises
            # over life.  SOH itself falls, so transform causal SOH loss first.
            raw_signal=(1.0 - series.soh).astype(np.float32),
            rul=series.rul.astype(np.float32),
            exact=series.target_mask.astype(bool),
            lower=series.lower_bound_rul.astype(np.float32),
            status=series.status,
            metadata={
                "protocol": series.protocol,
                "eol_ah": series.eol_ah,
                "observed_cycles": len(series),
                "life_cycle": series.life_cycle,
            },
        )
        manifest["files"][name] = {"path": str(path), "sha256": sha256_file(path), "status": series.status, "n_rows": len(series)}
    observed = tuple(name for name in BATTERY_EVENT_FOLDS if records[name].status == "event_observed")
    if observed != BATTERY_EVENT_FOLDS:
        raise ValueError(f"strict14 event-observed folds mismatch: {observed}")
    if records[BATTERY_CENSORED].status != "right_censored":
        raise ValueError("B0007 must remain right-censored under strict14")
    correlation = direction_correlation(
        np.concatenate([1.0 - records[name].raw_signal for name in BATTERY_EVENT_FOLDS]),
        np.concatenate([records[name].rul for name in BATTERY_EVENT_FOLDS]),
    )
    if not np.isfinite(correlation) or correlation <= 0.0:
        raise ValueError(f"Battery health/RUL direction check failed: correlation={correlation}")
    manifest["event_observed_folds"] = list(BATTERY_EVENT_FOLDS)
    manifest["censored_diagnostic"] = BATTERY_CENSORED
    manifest["target_input"] = "causal capacity damage = 1 - SOH; transform maps damage back to health"
    manifest["mean_health_rul_correlation"] = correlation
    return records, manifest


def load_rw_records(femto_zip: Path) -> tuple[dict[str, TargetRecord], dict[str, object]]:
    series_by_name = load_femto_zip(femto_zip, feature_group="base")
    names = tuple(series_by_name)
    if names != BEARING_NAMES:
        raise ValueError(f"Unexpected FEMTO bearing list: {names}")
    records: dict[str, TargetRecord] = {}
    correlations = []
    for name, series in series_by_name.items():
        # acc_x_rms rises with mechanical damage, so its inverse is health.
        health_proxy = -series.features[:, 0].astype(np.float32)
        correlation = direction_correlation(health_proxy, series.raw_rul)
        correlations.append(correlation)
        records[name] = TargetRecord(
            name=name,
            raw_signal=series.features[:, 0].astype(np.float32),
            rul=series.raw_rul.astype(np.float32),
            exact=np.ones(len(series), dtype=bool),
            lower=series.raw_rul.astype(np.float32),
            status="event_observed_ordinal_proxy",
            metadata={"n_records": len(series), "endpoint_rule": series.endpoint_rule, "health_rul_correlation": correlation},
        )
    correlation = float(np.nanmean(np.asarray(correlations, dtype=np.float64)))
    if not np.isfinite(correlation) or correlation <= 0.0:
        raise ValueError(f"FEMTO RMS health/RUL direction check failed: correlation={correlation}")
    return records, {
        "path": str(femto_zip),
        "sha256": sha256_file(femto_zip),
        "bearings": {name: records[name].metadata for name in BEARING_NAMES},
        "mean_health_rul_correlation": correlation,
        "target_label": "ordinal remaining measurements within complete FEMTO Learning_set run",
    }


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    return "cpu"


def pretrain_source(source: Mapping[str, PreparedData], out_dir: Path, *, device: str, epochs: int,
                    patience: int, batch_size: int, seeds: Sequence[int], heartbeat: Path | None = None) -> dict[str, object]:
    train = source["train"]
    validation = source.get("validation")
    source_test = source.get("source_test")
    checkpoint_dir = out_dir / "pretrained_encoder"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    for seed in seeds:
        checkpoint = checkpoint_dir / f"nozzle_encoder_seed{seed}.pt"
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            seed_reports.append(dict(payload.get("report", {})))
            continue
        seed_everything(int(seed))
        model = TargetRULModel().to(device)
        model, diag = train_model(
            model, train, validation, device=device, seed=int(seed), epochs=epochs, patience=patience,
            batch_size=batch_size, freeze_encoder=False, heartbeat=heartbeat,
            heartbeat_fields={"stage": "source_pretrain", "seed": int(seed)},
        )
        test_metric = None
        if source_test is not None and len(source_test):
            pred = predict_model(model, source_test, device, 1.0)
            test_metric = prediction_metrics(source_test.y[source_test.exact], pred[source_test.exact], 1.0)
        report = {"seed": int(seed), "training": diag, "source_test": test_metric}
        torch.save({"schema": SCHEMA, "encoder": model.encoder.state_dict(), "report": report}, checkpoint)
        seed_reports.append(report)
        write_heartbeat(heartbeat, stage="source_pretrain_complete", seed=int(seed))
    return {"checkpoint_dir": str(checkpoint_dir), "seed_reports": seed_reports}


def _load_encoder(model: TargetRULModel, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Pretraining checkpoint schema mismatch: {path}")
    model.encoder.load_state_dict(payload["encoder"])


def _censored_diagnostic(test: PreparedData, prediction: np.ndarray) -> dict[str, object]:
    lower = test.lower.astype(np.float64)
    prediction_norm = prediction.astype(np.float64)
    violations = np.maximum(lower - prediction_norm, 0.0)
    return {
        "n_points": int(len(test)),
        "mean_lower_bound_violation": float(np.mean(violations)),
        "max_lower_bound_violation": float(np.max(violations)),
        "violation_fraction": float(np.mean(violations > 1.0e-8)),
        "raw_rmse": None,
        "mae": None,
        "bias": None,
        "normalized_rmse": None,
    }


def _prediction_rows(cell: Mapping[str, Any], arm: str, prediction: np.ndarray, test: PreparedData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for endpoint, unit, truth, lower, exact, value in zip(
        test.endpoints, test.units, test.y, test.lower, test.exact, prediction
    ):
        rows.append({
            "domain": cell["domain"],
            "holdout": cell["holdout"],
            "n_shot": int(cell["n_shot"]),
            "trial": int(cell["trial"]),
            "arm": arm,
            "unit": str(unit),
            "endpoint": int(endpoint),
            "target_exact": bool(exact),
            "y_true_raw": float(truth * cell["label_scale"]) if bool(exact) else None,
            "lower_bound_raw": float(lower * cell["label_scale"]),
            "prediction_raw": float(value),
        })
    return rows


def execute_cell(payload: Mapping[str, Any]) -> dict[str, object]:
    root = Path(payload["root"])
    output = Path(payload["output"])
    domain = str(payload["domain"])
    holdout = str(payload["holdout"])
    selected = tuple(str(value) for value in payload["selected"])
    n_shot = int(payload["n_shot"])
    trial = int(payload["trial"])
    device = resolve_device(str(payload["device"]))
    heartbeat = Path(payload["heartbeat"])
    write_heartbeat(heartbeat, stage="loading", domain=domain, holdout=holdout, n_shot=n_shot, trial=trial)
    if domain in ("battery", "battery_censored"):
        records, _ = load_battery_records(Path(payload["battery_dir"]))
        seq_len = SEQ_LEN["battery"]
    elif domain == "rw_proxy":
        records, _ = load_rw_records(Path(payload["femto_zip"]))
        seq_len = SEQ_LEN["rw_proxy"]
    else:
        raise ValueError(f"Unknown target domain: {domain}")
    bundle = build_target_bundle(records, selected=selected, holdout=holdout, seq_len=seq_len)
    if not len(bundle.test):
        raise ValueError("Holdout has no test windows")
    if not np.all(np.isfinite(bundle.train.x)) or not np.all(np.isfinite(bundle.test.x)):
        raise FloatingPointError("Non-finite target health input")
    cell_common = {
        "domain": domain,
        "holdout": holdout,
        "selected": list(selected),
        "n_shot": n_shot,
        "trial": trial,
        "label_scale": float(bundle.label_scale),
    }
    results: dict[str, object] = {}
    prediction_rows: list[dict[str, object]] = []
    pretrained_dir = output / "pretrained_encoder"
    neural_arms = (("transfer", False), ("frozen_encoder_ablation", True), ("scratch", False))
    for arm, freeze_encoder in neural_arms:
        per_seed, seed_predictions = [], []
        for seed in payload["seeds"]:
            seed = int(seed)
            write_heartbeat(heartbeat, stage="training", arm=arm, seed=seed, **cell_common)
            seed_everything(seed)
            model = TargetRULModel().to(device)
            if arm != "scratch":
                _load_encoder(model, pretrained_dir / f"nozzle_encoder_seed{seed}.pt")
            model, diag = train_model(
                model, bundle.train, bundle.val, device=device, seed=seed, epochs=int(payload["target_epochs"]),
                patience=int(payload["patience"]), batch_size=int(payload["batch_size"]), freeze_encoder=freeze_encoder,
                heartbeat=heartbeat, heartbeat_fields={"arm": arm, "seed": seed, **cell_common},
            )
            prediction = predict_model(model, bundle.test, device, bundle.label_scale)
            seed_predictions.append(prediction)
            if np.any(bundle.test.exact):
                metric = prediction_metrics(
                    bundle.test.y[bundle.test.exact] * bundle.label_scale,
                    prediction[bundle.test.exact],
                    bundle.label_scale,
                )
            else:
                metric = _censored_diagnostic(bundle.test, prediction / bundle.label_scale)
            per_seed.append({"seed": seed, "metrics": metric, "training": diag})
        ensemble = np.mean(np.stack(seed_predictions, axis=0), axis=0).astype(np.float32)
        if np.any(bundle.test.exact):
            metric = prediction_metrics(
                bundle.test.y[bundle.test.exact] * bundle.label_scale,
                ensemble[bundle.test.exact],
                bundle.label_scale,
            )
        else:
            metric = _censored_diagnostic(bundle.test, ensemble / bundle.label_scale)
        results[arm] = {"metrics": metric, "seed_results": per_seed}
        prediction_rows.extend(_prediction_rows(cell_common, arm, ensemble, bundle.test))
    ridge_prediction = predict_ridge(fit_ridge(bundle.train), bundle.test, bundle.label_scale)
    trend_prediction = predict_time_trend(fit_time_trend(bundle.train, bundle.label_scale), bundle.test, bundle.label_scale)
    for arm, prediction in (("ridge", ridge_prediction), ("time_trend", trend_prediction)):
        metric = prediction_metrics(
            bundle.test.y[bundle.test.exact] * bundle.label_scale,
            prediction[bundle.test.exact],
            bundle.label_scale,
        ) if np.any(bundle.test.exact) else _censored_diagnostic(bundle.test, prediction / bundle.label_scale)
        results[arm] = {"metrics": metric}
        prediction_rows.extend(_prediction_rows(cell_common, arm, prediction, bundle.test))
    write_heartbeat(heartbeat, stage="completed", **cell_common)
    return {
        "schema": SCHEMA,
        "status": "completed",
        **cell_common,
        "target_health_transform": bundle.transform.as_dict(),
        "target_split": bundle.split_audit,
        "results": results,
        "predictions": prediction_rows,
    }


def worker_main(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    result_path = Path(payload["result_path"])
    heartbeat = Path(payload["heartbeat"])
    try:
        result = execute_cell(payload)
        atomic_write_json(result_path, result)
    except Exception as exc:
        write_heartbeat(heartbeat, stage="failed", error=repr(exc))
        atomic_write_json(result_path, {"schema": SCHEMA, "status": "failed", "error": repr(exc), "payload": payload})
        raise


def protocol_payload(args: argparse.Namespace, *, smoke: bool) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "source": {
            "name": "strict 200 multi-trajectory nozzle ablation",
            "input": "health = 1 - normalized_depth, normalized per source trajectory",
            "split": "trajectory train/validation/source-test",
            "pretraining_selection": "source validation only; source test audit only",
        },
        "targets": {
            "battery": {
                "protocol": "strict14",
                "event_observed_outer_folds": list(BATTERY_EVENT_FOLDS),
                "right_censored_diagnostic": BATTERY_CENSORED,
                "score_target": "raw remaining discharge cycles",
            },
            "rw_proxy": {
                "protocol": "six-bearing outer LOO",
                "target": "FEMTO mechanical degradation proxy, ordinal remaining measurements",
                "rms_calibration": "fine-tuning training-prefix RMS only; no holdout or full-life min/max",
            },
        },
        "model": {
            "source_target_input_projection": "domain-specific and independently initialized",
            "shared_component": "two-layer GRU degradation encoder",
            "target_head": "independently initialized",
            "main_transfer": "pretrained encoder, target projection, and head all fine-tuned",
            "ablation": "pretrained frozen encoder",
            "cycle_pos": "removed from main protocol",
        },
        "fairness": {
            "arms": ["transfer", "scratch", "frozen_encoder_ablation", "ridge", "time_trend"],
            "neural_target_epoch_budget": int(args.target_epochs),
            "early_stopping": "same chronological validation for transfer, scratch, and frozen ablation",
            "seeds": list(args.seeds),
            "trial_selection": "SHA-256-derived explicit seed, not Python hash()",
        },
        "run": {
            "smoke": bool(smoke),
            "source_epochs": int(args.source_epochs),
            "target_epochs": int(args.target_epochs),
            "seeds": list(args.seeds),
            "trials_per_n": int(args.trials),
            "early_frac": EARLY_FRAC,
        },
        "acceptance": {
            "minimum_macro_rmse_improvement_pct": 5.0,
            "majority_outer_folds_better_than_scratch": True,
            "single_fold_only_is_not_positive_evidence": True,
            "battery_and_rw_proxy_assessed_separately": True,
        },
    }


def preflight(args: argparse.Namespace, output: Path, *, smoke: bool) -> tuple[dict[str, object], dict[str, PreparedData]]:
    nozzle_path = Path(args.nozzle_data)
    battery_dir = Path(args.battery_dir)
    femto_zip = Path(args.femto_zip)
    missing = [str(path) for path in (nozzle_path, battery_dir, femto_zip) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"v3 preflight missing data: {missing}")
    source, source_manifest = load_source_split(nozzle_path, source_limit=(2 if smoke else None))
    _, battery_manifest = load_battery_records(battery_dir)
    _, rw_manifest = load_rw_records(femto_zip)
    manifest = {
        "schema": f"{SCHEMA}_data_manifest",
        "created_at_utc": utc_now(),
        "source_nozzle": source_manifest,
        "battery": battery_manifest,
        "rw_proxy": rw_manifest,
        "integrity_checks": {
            "source_trajectory_ids_unique": True,
            "source_expected_trajectory_count": 200 if not smoke else source_manifest["trajectory_count"],
            "battery_strict14_outer_fold_status": "verified_event_observed",
            "battery_b0007_status": "verified_right_censored",
            "rw_expected_bearing_count": 6,
            "health_rul_direction": "verified_positive_correlation_in_each_domain",
            "hashes": "SHA-256 recorded; re-run compares to this manifest before resume",
        },
    }
    previous = output / "DATA_MANIFEST.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        old_hashes = {
            "source": old.get("source_nozzle", {}).get("sha256"),
            "femto": old.get("rw_proxy", {}).get("sha256"),
            "battery": {key: item.get("sha256") for key, item in old.get("battery", {}).get("files", {}).items()},
        }
        new_hashes = {
            "source": source_manifest["sha256"],
            "femto": rw_manifest["sha256"],
            "battery": {key: item["sha256"] for key, item in battery_manifest["files"].items()},
        }
        if old_hashes != new_hashes:
            raise ValueError("Existing DATA_MANIFEST hashes disagree with current inputs; refuse to resume")
        manifest["integrity_checks"]["resume_hash_comparison"] = "passed"
    else:
        manifest["integrity_checks"]["resume_hash_comparison"] = "initial_manifest_recorded"
    return manifest, source


def build_cells(args: argparse.Namespace, *, smoke: bool) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    domains: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = [
        ("battery", BATTERY_EVENT_FOLDS, N_SHOTS["battery"]),
        ("rw_proxy", BEARING_NAMES, N_SHOTS["rw_proxy"]),
    ]
    if smoke:
        domains = [("battery", (BATTERY_EVENT_FOLDS[0],), (1,)), ("rw_proxy", (BEARING_NAMES[0],), (1,))]
    for domain, holdouts, shots in domains:
        universe = BATTERY_EVENT_FOLDS if domain == "battery" else BEARING_NAMES
        for holdout in holdouts:
            available = [name for name in universe if name != holdout]
            for n_shot in shots:
                for trial in range(args.trials if not smoke else 1):
                    selected = select_trial_units(available, n_shot, trial, domain=domain, holdout=holdout)
                    cells.append({"domain": domain, "holdout": holdout, "n_shot": n_shot, "trial": trial, "selected": selected})
    # B0007 is never placed into scored event-observed battery macro metrics.
    if not smoke:
        for n_shot in N_SHOTS["battery"]:
            for trial in range(args.trials):
                selected = select_trial_units(BATTERY_EVENT_FOLDS, n_shot, trial, domain="battery_censored", holdout=BATTERY_CENSORED)
                cells.append({"domain": "battery_censored", "holdout": BATTERY_CENSORED, "n_shot": n_shot, "trial": trial, "selected": selected})
    return cells


def _cell_id(cell: Mapping[str, object]) -> str:
    return f"{cell['domain']}__{cell['holdout']}__N{cell['n_shot']}__trial{cell['trial']}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError):
        process.terminate()
    deadline = time.monotonic() + 10.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError):
            process.kill()


def run_supervisor(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.smoke)
    protocol = protocol_payload(args, smoke=smoke)
    atomic_write_json(output / "PROTOCOL.json", protocol)
    status_path = output / "SUPERVISOR_STATUS.json"
    write_heartbeat(status_path, schema=SCHEMA, status="preflight", smoke=smoke)
    manifest, source = preflight(args, output, smoke=smoke)
    atomic_write_json(output / "DATA_MANIFEST.json", manifest)
    device = resolve_device(args.device)
    write_heartbeat(status_path, schema=SCHEMA, status="source_pretraining", device=device, smoke=smoke)
    pretraining = pretrain_source(
        source, output, device=device, epochs=args.source_epochs, patience=args.patience,
        batch_size=args.batch_size, seeds=args.seeds, heartbeat=status_path,
    )
    cells = build_cells(args, smoke=smoke)
    checkpoint_path = output / "CROSS_TRANSFER_V3_CHECKPOINT.json"
    checkpoint = _read_json(checkpoint_path) or {"schema": SCHEMA, "cells": {}, "failures": []}
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("Existing v3 checkpoint has a different schema")
    started = time.monotonic()
    total = len(cells)
    for index, cell in enumerate(cells, start=1):
        key = _cell_id(cell)
        existing = checkpoint["cells"].get(key)
        if existing and existing.get("status") == "completed":
            continue
        cell_dir = output / "cells" / key
        cell_dir.mkdir(parents=True, exist_ok=True)
        payload_path = cell_dir / "payload.json"
        result_path = cell_dir / "result.json"
        heartbeat_path = cell_dir / "heartbeat.json"
        payload = {
            **cell,
            "root": str(ROOT),
            "output": str(output),
            "result_path": str(result_path),
            "heartbeat": str(heartbeat_path),
            "device": device,
            "nozzle_data": str(args.nozzle_data),
            "battery_dir": str(args.battery_dir),
            "femto_zip": str(args.femto_zip),
            "seeds": list(args.seeds),
            "target_epochs": int(args.target_epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
        }
        atomic_write_json(payload_path, payload)
        attempt = 0
        completed = False
        while attempt <= args.retries and not completed:
            attempt += 1
            result_path.unlink(missing_ok=True)
            write_heartbeat(status_path, schema=SCHEMA, status="running", cell=key, cell_index=index, total_cells=total,
                            attempt=attempt, elapsed_seconds=round(time.monotonic() - started, 1))
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker-json", str(payload_path)],
                cwd=str(ROOT), start_new_session=True,
            )
            launch = time.monotonic()
            last_heartbeat_wall = time.time()
            while process.poll() is None:
                time.sleep(2.0)
                record = _read_json(heartbeat_path)
                if record is not None:
                    try:
                        last_heartbeat_wall = max(last_heartbeat_wall, heartbeat_path.stat().st_mtime)
                    except OSError:
                        pass
                elapsed = time.monotonic() - launch
                stale = time.time() - last_heartbeat_wall
                if elapsed > args.cell_timeout or stale > args.stall_timeout:
                    _terminate(process)
                    checkpoint["failures"].append({
                        "cell": key, "attempt": attempt, "reason": "hard_timeout" if elapsed > args.cell_timeout else "heartbeat_stall",
                        "elapsed_seconds": elapsed, "stale_seconds": stale,
                    })
                    break
            result = _read_json(result_path)
            if process.returncode == 0 and result is not None and result.get("status") == "completed":
                checkpoint["cells"][key] = result
                completed = True
            else:
                checkpoint["failures"].append({
                    "cell": key, "attempt": attempt, "reason": "worker_failure",
                    "returncode": process.returncode, "result": result,
                })
            checkpoint["updated_at_utc"] = utc_now()
            checkpoint["pretraining"] = pretraining
            atomic_write_json(checkpoint_path, checkpoint)
        if not completed:
            write_heartbeat(status_path, schema=SCHEMA, status="failed", failed_cell=key,
                            completed_cells=len(checkpoint["cells"]), total_cells=total, failures=checkpoint["failures"])
            raise RuntimeError(f"v3 worker did not complete after retries: {key}")
    final = build_final_report(checkpoint["cells"], protocol, manifest, pretraining, smoke=smoke)
    atomic_write_json(output / "CROSS_TRANSFER_V3_REPORT.json", final)
    write_final_csvs(output, checkpoint["cells"], final)
    write_heartbeat(status_path, schema=SCHEMA, status="completed", completed_cells=len(checkpoint["cells"]),
                    total_cells=total, smoke=smoke, report=str(output / "CROSS_TRANSFER_V3_REPORT.json"))


def _mean_metric(items: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None and np.isfinite(float(item[key]))]
    return float(np.mean(values)) if values else None


def build_final_report(cells: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any],
                       manifest: Mapping[str, Any], pretraining: Mapping[str, Any], *, smoke: bool) -> dict[str, object]:
    by_domain: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells.values():
        by_domain.setdefault(str(cell["domain"]), []).append(cell)
    macro_rows = []
    fold_rows = []
    acceptance: dict[str, object] = {}
    for domain in ("battery", "rw_proxy"):
        domain_cells = by_domain.get(domain, [])
        shots = sorted({int(item["n_shot"]) for item in domain_cells})
        domain_acceptance = []
        for n_shot in shots:
            subset = [item for item in domain_cells if int(item["n_shot"]) == n_shot]
            arms = sorted({arm for item in subset for arm in item["results"]})
            by_fold: dict[str, list[Mapping[str, Any]]] = {}
            for item in subset:
                by_fold.setdefault(str(item["holdout"]), []).append(item)
            fold_arm_metrics: dict[str, dict[str, dict[str, float | None]]] = {}
            for holdout, fold_cells in sorted(by_fold.items()):
                fold_arm_metrics[holdout] = {}
                for arm in arms:
                    metrics = [item["results"][arm]["metrics"] for item in fold_cells]
                    summary = {key: _mean_metric(metrics, key) for key in ("raw_rmse", "mae", "bias", "normalized_rmse")}
                    fold_arm_metrics[holdout][arm] = summary
                    fold_rows.append({"domain": domain, "n_shot": n_shot, "holdout": holdout, "arm": arm, **summary,
                                      "n_trials": len(metrics)})
            for arm in arms:
                metrics = [fold_arm_metrics[fold][arm] for fold in fold_arm_metrics]
                macro = {key: _mean_metric(metrics, key) for key in ("raw_rmse", "mae", "bias", "normalized_rmse")}
                scratch = {key: _mean_metric([fold_arm_metrics[fold]["scratch"] for fold in fold_arm_metrics], key)
                           for key in ("raw_rmse", "mae", "bias", "normalized_rmse")} if "scratch" in arms else {}
                gain = None
                folds_better = None
                if arm == "transfer" and scratch.get("raw_rmse") is not None and macro["raw_rmse"] is not None:
                    gain = (scratch["raw_rmse"] - macro["raw_rmse"]) / max(scratch["raw_rmse"], 1.0e-8) * 100.0
                    folds_better = sum(
                        fold_arm_metrics[fold]["transfer"]["raw_rmse"] < fold_arm_metrics[fold]["scratch"]["raw_rmse"]
                        for fold in fold_arm_metrics
                    )
                macro_rows.append({
                    "domain": domain, "n_shot": n_shot, "arm": arm, **macro,
                    "n_outer_folds": len(fold_arm_metrics), "folds_better_than_scratch": folds_better,
                    "gain_pct_vs_scratch": gain,
                })
            transfer_row = next((row for row in macro_rows if row["domain"] == domain and row["n_shot"] == n_shot and row["arm"] == "transfer"), None)
            accepted = bool(
                transfer_row
                and transfer_row["gain_pct_vs_scratch"] is not None
                and transfer_row["gain_pct_vs_scratch"] >= 5.0
                and transfer_row["folds_better_than_scratch"] > transfer_row["n_outer_folds"] / 2
            )
            domain_acceptance.append({
                "n_shot": n_shot,
                "positive_transfer_evidence": accepted,
                "reason": "macro >=5% and majority folds better" if accepted else "diagnostic: acceptance threshold not met",
            })
        acceptance[domain] = {
            "per_n_shot": domain_acceptance,
            "overall_positive_transfer_evidence": any(item["positive_transfer_evidence"] for item in domain_acceptance),
            "rule": "each domain is judged separately; any failure remains diagnostic rather than a success claim",
        }
    censored = by_domain.get("battery_censored", [])
    censored_summary = []
    for item in censored:
        for arm, value in item["results"].items():
            censored_summary.append({
                "holdout": item["holdout"], "n_shot": item["n_shot"], "trial": item["trial"], "arm": arm,
                **value["metrics"],
            })
    return {
        "schema": SCHEMA,
        "status": "smoke_complete" if smoke else "complete",
        "generated_at_utc": utc_now(),
        "protocol": protocol,
        "data_manifest": manifest,
        "source_pretraining": pretraining,
        "fold_metrics": fold_rows,
        "macro_metrics": macro_rows,
        "acceptance": acceptance,
        "battery_b0007_right_censored_diagnostic": censored_summary,
        "positive_claim_rule": "Only domains/N-shot settings meeting every stated acceptance condition are positive evidence.",
    }


def write_final_csvs(output: Path, cells: Mapping[str, Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    macro_fields = [
        "domain", "n_shot", "arm", "raw_rmse", "mae", "bias", "normalized_rmse",
        "n_outer_folds", "folds_better_than_scratch", "gain_pct_vs_scratch",
    ]
    atomic_write_csv(output / "CROSS_TRANSFER_V3_MACRO.csv", report["macro_metrics"], macro_fields)
    rows = [row for cell in cells.values() for row in cell.get("predictions", [])]
    prediction_fields = [
        "domain", "holdout", "n_shot", "trial", "arm", "unit", "endpoint",
        "target_exact", "y_true_raw", "lower_bound_raw", "prediction_raw",
    ]
    atomic_write_csv(output / "CROSS_TRANSFER_V3_PREDICTIONS.csv", rows, prediction_fields)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/cross_component_transfer_v3")
    parser.add_argument("--nozzle-data", default="data/processed/nozzle_mt_200/nozzle_sim_200traj.csv")
    parser.add_argument("--battery-dir", default="data/processed/nasa_battery/5. Battery Data Set")
    parser.add_argument("--femto-zip", default="data/processed/femto_bearing.zip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-epochs", type=int, default=DEFAULT_SOURCE_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=DEFAULT_TARGET_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--cell-timeout", type=float, default=7200.0)
    parser.add_argument("--stall-timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--worker-json", type=Path)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.worker_json is not None:
        worker_main(args.worker_json)
        return
    if args.smoke:
        args.source_epochs = min(args.source_epochs, 18)
        args.target_epochs = min(args.target_epochs, 24)
        args.patience = min(args.patience, 6)
        args.seeds = list(args.seeds[:2])
        args.cell_timeout = min(args.cell_timeout, 900.0)
        args.stall_timeout = min(args.stall_timeout, 180.0)
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    run_supervisor(args)


if __name__ == "__main__":
    main()
