#!/usr/bin/env python3
"""Leakage-safe two-stage benchmark for the strict nozzle trajectory.

Selection uses only fold1/fold2 train and validation partitions.  Final
execution requires the resulting ``frozen_config.json`` and evaluates every
predeclared model, seed, fold, baseline, and ablation without test-driven
checkpoint, seed, family, or ensemble-weight selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.nozzle_strict import (
    CANONICAL_SHA256,
    FEATURE_TIERS,
    MAIN_FEATURE_TIER,
    NozzleData,
    PreparedFold,
    WindowSet,
    get_split_specs,
    load_nozzle_csv,
    prepare_fold,
)
from src.models.nozzle_strict import build_nozzle_model


SCHEMA_VERSION = "nozzle-strict-experiment-v1"
DEFAULT_SEEDS = (42, 123, 456, 2026, 3407)
MODEL_FAMILIES = ("lstm", "tcn", "transformer", "physics_input", "physics_full")
PRIMARY_MODEL = "physics_full"
FEATURE_WINDOW_ABLATION_MODEL = "tcn"
PRIMARY_FEATURE_TIER = MAIN_FEATURE_TIER
PRIMARY_WINDOW_SIZE = 3
SELECTION_FOLDS = ("fold1", "fold2")
FINAL_FOLDS = ("fold1", "fold2", "fold3")
LOCKED_FOLD = "fold3"
FAILURE_TIME_S = 20.0
QUICK_MAX_EPOCHS = 10
QUICK_PATIENCE = 3
METRIC_NAMES = ("rmse", "mae", "weighted_rmse", "weighted_mae", "nrmse")
LINEAR_ALPHAS = (0.0, 1.0e-4, 1.0e-2, 1.0, 100.0)
ELIGIBLE_BASELINES = ("current_rate", "local_slope", "ridge", "huber")
DIAGNOSTIC_METHODS = ("clock_oracle",)

# These are intentionally module constants: selection can choose only among
# this small, predeclared set.  Quick mode caps execution epochs but does not
# alter or expand the candidate set.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "compact",
        "hidden": 16,
        "dropout": 0.05,
        "lr": 3.0e-3,
        "weight_decay": 1.0e-4,
        "patience": 16,
        "max_epochs": 140,
    },
    {
        "id": "balanced",
        "hidden": 24,
        "dropout": 0.10,
        "lr": 1.5e-3,
        "weight_decay": 1.0e-4,
        "patience": 25,
        "max_epochs": 220,
    },
    {
        "id": "regularized",
        "hidden": 32,
        "dropout": 0.15,
        "lr": 8.0e-4,
        "weight_decay": 1.0e-3,
        "patience": 35,
        "max_epochs": 300,
    },
)

PROTOCOL_SPEC: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "trajectory_protocol": "chronological expanding-origin; raw-disjoint train/val/test",
    "selection": {
        "folds": list(SELECTION_FOLDS),
        "partitions_available_to_trainer": ["train", "val"],
        "selection_model": PRIMARY_MODEL,
        "feature_tier": PRIMARY_FEATURE_TIER,
        "window_size": PRIMARY_WINDOW_SIZE,
        "objective": "macro mean validation RMSE; validation weighted RMSE then candidate id tie-break",
        "seed_policy": "first preregistered seed only; no seed selection",
    },
    "final": {
        "folds": list(FINAL_FOLDS),
        "locked_fold": LOCKED_FOLD,
        "model_families": list(MODEL_FAMILIES),
        "primary_model": PRIMARY_MODEL,
        "feature_tier": PRIMARY_FEATURE_TIER,
        "window_size": PRIMARY_WINDOW_SIZE,
        "checkpoint_selection": "validation RMSE only, including an epoch-0 checkpoint",
        "test_policy": "one prediction pass after checkpoint lock; no seed/family selection",
        "ensemble": "fixed equal prediction average over all preregistered effective seeds",
    },
    "loss": {
        "all_models": "scaled weighted smooth_l1 RUL",
        "physics_input": "RUL term only",
        "physics_full": [
            "RUL smooth_l1",
            "observed current-rate supervision after m/s to mm/s conversion",
            "predicted_rate_mm_s * predicted_rul_s versus depth_margin_mm consistency",
        ],
        "rate_weight": 0.20,
        "consistency_weight": 0.20,
        "gradient_clip_norm": 5.0,
        "scale_scope": "train partition only",
    },
    "baselines": {
        "clock_oracle": "20-t diagnostic; excluded from ranking",
        "current_rate": "observed current-rate depth-margin extrapolation",
        "local_slope": "least-squares depth slope over each partition-local window",
        "ridge": "flattened windows; train fit; validation-only alpha choice",
        "huber": "flattened windows; train IRLS fit; validation-only alpha choice",
        "linear_alphas": list(LINEAR_ALPHAS),
    },
    "ablations": {
        "feature_tiers": list(FEATURE_TIERS),
        "window_sizes": [1, 3, 5],
        "feature_window_model": FEATURE_WINDOW_ABLATION_MODEL,
        "feature_window_model_reason": "isolates input/window effects without an external physics-RUL branch",
        "physics_models": ["tcn", "physics_input", "physics_full"],
        "age_aware": "diagnostic and excluded from ranking",
    },
    "metrics": {
        "names": list(METRIC_NAMES),
        "weights": "partition-local normalized trapezoidal WindowSet.weights",
        "nrmse_denominator": FAILURE_TIME_S,
        "aggregation": "single-seed mean/std plus fixed equal-prediction ensemble; fold macro is unweighted",
    },
    "claim_rule": {
        "maximum_text": "task-specific strict best on this single trajectory",
        "requirements": [
            "preregistered primary trainable model beats every eligible baseline on locked-fold RMSE",
            "preregistered primary trainable model beats every eligible baseline on rolling-fold macro RMSE",
            "primary single-seed model beats every eligible baseline in a strict majority of seed-fold cells",
        ],
        "global_sota_allowed": False,
    },
}


@dataclass
class NeuralRun:
    experiment: str
    model: str
    fold: str
    seed: int
    feature_tier: str
    window_size: int
    metrics: dict[str, float]
    val_best: dict[str, float]
    best_epoch: int
    epochs_ran: int
    parameter_count: int
    training_seconds: float
    inference_seconds: float
    loss_scales: dict[str, float]
    endpoint_indices: np.ndarray
    endpoint_times_s: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    weights: np.ndarray
    quadrature_weights_s: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "model": self.model,
            "fold": self.fold,
            "seed": self.seed,
            "feature_tier": self.feature_tier,
            "window_size": self.window_size,
            "metrics": self.metrics,
            "val_best": self.val_best,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "parameter_count": self.parameter_count,
            "training_seconds": self.training_seconds,
            "inference_seconds": self.inference_seconds,
            "loss_scales": self.loss_scales,
        }


@dataclass
class BaselineRun:
    model: str
    fold: str
    diagnostic: bool
    ranking_eligible: bool
    metrics: dict[str, float]
    val_metrics: dict[str, float]
    selected_alpha: float | None
    training_seconds: float
    inference_seconds: float
    endpoint_indices: np.ndarray
    endpoint_times_s: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    weights: np.ndarray
    quadrature_weights_s: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "fold": self.fold,
            "diagnostic": self.diagnostic,
            "ranking_eligible": self.ranking_eligible,
            "metrics": self.metrics,
            "val_metrics": self.val_metrics,
            "selected_alpha": self.selected_alpha,
            "training_seconds": self.training_seconds,
            "inference_seconds": self.inference_seconds,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_text(path, text)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _write_text(path, "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})
    temporary.replace(path)


def parse_seeds(text: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in text.replace(";", ",").split(",")]
    try:
        seeds = tuple(int(token) for token in tokens if token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise argparse.ArgumentTypeError("each seed must be in [0, 2^32-1]")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


def resolve_device(requested: str) -> tuple[torch.device, str | None]:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), None
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu"), f"requested {requested!r}, but CUDA is unavailable; using CPU"
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"requested CUDA device index {index}, but only {torch.cuda.device_count()} device(s) exist"
            )
        return torch.device(f"cuda:{index}"), None
    if device.type != "cpu":
        raise ValueError("--device must be auto, cpu, cuda, or cuda:<index>")
    return device, None


def configure_runtime() -> None:
    # Tiny full-batch models are faster without a large CPU thread pool.
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def effective_config(config: Mapping[str, Any], quick: bool) -> dict[str, Any]:
    result = dict(config)
    if quick:
        result["max_epochs"] = min(int(result["max_epochs"]), QUICK_MAX_EPOCHS)
        result["patience"] = min(int(result["patience"]), QUICK_PATIENCE)
    return result


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    weights: Sequence[float],
    failure_time_s: float,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.maximum(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0)
    sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (len(truth) == len(prediction) == len(sample_weights)) or len(truth) == 0:
        raise ValueError("metric arrays must be non-empty and have matching lengths")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metrics received non-finite labels or predictions")
    if np.any(sample_weights < 0.0) or not np.isfinite(sample_weights).all():
        raise ValueError("metric weights must be finite and non-negative")
    total_weight = float(sample_weights.sum())
    if total_weight <= 0.0:
        raise ValueError("metric weights must have positive total")
    normalized_weights = sample_weights / total_weight
    errors = prediction - truth
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    weighted_rmse = float(np.sqrt(np.sum(normalized_weights * np.square(errors))))
    weighted_mae = float(np.sum(normalized_weights * np.abs(errors)))
    denominator = max(float(failure_time_s), np.finfo(np.float64).eps)
    return {
        "rmse": rmse,
        "mae": mae,
        "weighted_rmse": weighted_rmse,
        "weighted_mae": weighted_mae,
        "nrmse": rmse / denominator,
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    safe_weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return torch.sum(values * safe_weights)


def _loss_scales(train_set: WindowSet) -> dict[str, float]:
    observed_rate_mm_s = np.asarray(train_set.current_rate_m_s, dtype=np.float64) * 1000.0
    return {
        "rul_s": max(float(np.median(np.abs(train_set.y_rul_s))), 1.0),
        "rate_mm_s": max(float(np.median(np.abs(observed_rate_mm_s))), 1.0e-5),
        "depth_margin_mm": max(float(np.median(np.abs(train_set.depth_margin_mm))), 1.0e-4),
    }


def _model_loss(
    model_name: str,
    output: Any,
    y_rul: torch.Tensor,
    observed_rate_mm_s: torch.Tensor,
    depth_margin_mm: torch.Tensor,
    weights: torch.Tensor,
    scales: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_rul = output.rul.clamp_min(0.0)
    rul_elementwise = F.smooth_l1_loss(
        predicted_rul / float(scales["rul_s"]),
        y_rul / float(scales["rul_s"]),
        reduction="none",
        beta=0.10,
    )
    rul_loss = _weighted_mean(rul_elementwise, weights)
    total = rul_loss
    pieces = {"rul": float(rul_loss.detach().cpu())}

    if model_name == "physics_full":
        if output.rate_mm_s is None:
            raise RuntimeError("physics_full did not expose rate_mm_s")
        predicted_rate = output.rate_mm_s.clamp_min(torch.finfo(output.rul.dtype).eps)
        rate_elementwise = F.smooth_l1_loss(
            predicted_rate / float(scales["rate_mm_s"]),
            observed_rate_mm_s / float(scales["rate_mm_s"]),
            reduction="none",
            beta=0.10,
        )
        consistency_elementwise = F.smooth_l1_loss(
            (predicted_rate * predicted_rul) / float(scales["depth_margin_mm"]),
            depth_margin_mm / float(scales["depth_margin_mm"]),
            reduction="none",
            beta=0.10,
        )
        rate_loss = _weighted_mean(rate_elementwise, weights)
        consistency_loss = _weighted_mean(consistency_elementwise, weights)
        total = total + 0.20 * rate_loss + 0.20 * consistency_loss
        pieces.update(
            {
                "rate": float(rate_loss.detach().cpu()),
                "consistency": float(consistency_loss.detach().cpu()),
            }
        )
    return total, pieces


def _partition_tensors(window_set: WindowSet, device: torch.device) -> dict[str, torch.Tensor]:
    # WindowSet arrays are deliberately read-only; copy before handing them to
    # torch so tensor construction cannot expose an unsafe writable view.
    return {
        "X": torch.as_tensor(np.array(window_set.X, copy=True), dtype=torch.float32, device=device),
        "y": torch.as_tensor(np.array(window_set.y_rul_s, copy=True), dtype=torch.float32, device=device),
        "physics": torch.as_tensor(
            np.array(window_set.physics_rul_s, copy=True), dtype=torch.float32, device=device
        ),
        "rate_mm_s": torch.as_tensor(
            np.array(window_set.current_rate_m_s, copy=True) * 1000.0,
            dtype=torch.float32,
            device=device,
        ),
        "margin_mm": torch.as_tensor(
            np.array(window_set.depth_margin_mm, copy=True), dtype=torch.float32, device=device
        ),
        "weights": torch.as_tensor(
            np.array(window_set.weights, copy=True), dtype=torch.float32, device=device
        ),
    }


def _forward_rul(model: torch.nn.Module, model_name: str, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    physics = tensors["physics"] if model_name.startswith("physics") else None
    return model(tensors["X"], physics).rul.clamp_min(0.0)


def train_model(
    model_name: str,
    n_features: int,
    train_set: WindowSet,
    val_set: WindowSet,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    failure_time_s: float,
    quick: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fit using train and validation only.

    Deliberately, this API has no test argument.  It seeds before constructing
    the model and returns a validation-selected checkpoint.
    """

    del failure_time_s  # Labels define metrics; fixed train-only scales define loss.
    run_config = effective_config(config, quick)
    seed_everything(seed)
    _sync(device)
    started = time.perf_counter()
    model = build_nozzle_model(
        model_name,
        n_features=n_features,
        hidden=int(run_config["hidden"]),
        dropout=float(run_config["dropout"]),
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    train_tensors = _partition_tensors(train_set, device)
    val_tensors = _partition_tensors(val_set, device)
    scales = _loss_scales(train_set)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["lr"]),
        weight_decay=float(run_config["weight_decay"]),
    )

    model.eval()
    with torch.no_grad():
        initial_val = _forward_rul(model, model_name, val_tensors).cpu().numpy()
    initial_metrics = compute_metrics(
        val_set.y_rul_s, initial_val, val_set.weights, FAILURE_TIME_S
    )
    if not math.isfinite(initial_metrics["rmse"]):
        raise RuntimeError(f"non-finite initial validation RMSE for {model_name}")
    best_rmse = initial_metrics["rmse"]
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    epochs_ran = 0
    last_loss: dict[str, float] = {}

    for epoch in range(1, int(run_config["max_epochs"]) + 1):
        epochs_ran = epoch
        model.train()
        physics = train_tensors["physics"] if model_name.startswith("physics") else None
        output = model(train_tensors["X"], physics)
        loss, last_loss = _model_loss(
            model_name,
            output,
            train_tensors["y"],
            train_tensors["rate_mm_s"],
            train_tensors["margin_mm"],
            train_tensors["weights"],
            scales,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss for {model_name} at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = _forward_rul(model, model_name, val_tensors).cpu().numpy()
        val_metrics = compute_metrics(
            val_set.y_rul_s, val_prediction, val_set.weights, FAILURE_TIME_S
        )
        if val_metrics["rmse"] < best_rmse - 1.0e-7:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(run_config["patience"]):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        selected_val_prediction = _forward_rul(model, model_name, val_tensors).cpu().numpy()
    selected_val_metrics = compute_metrics(
        val_set.y_rul_s, selected_val_prediction, val_set.weights, FAILURE_TIME_S
    )
    _sync(device)
    training_seconds = time.perf_counter() - started
    return model, {
        "best_val_metrics": selected_val_metrics,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "loss_scales": scales,
        "last_training_loss_terms": last_loss,
        "effective_config": run_config,
    }


def predict_partition(
    model: torch.nn.Module,
    model_name: str,
    window_set: WindowSet,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run one timed inference pass on an already locked checkpoint."""

    tensors = _partition_tensors(window_set, device)
    model.eval()
    _sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        prediction = _forward_rul(model, model_name, tensors).cpu().numpy()
    _sync(device)
    elapsed = time.perf_counter() - started
    prediction = np.maximum(np.asarray(prediction, dtype=np.float64), 0.0)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite inference output for {model_name}")
    return prediction, elapsed


def _clock_prediction(data: NozzleData, window_set: WindowSet) -> np.ndarray:
    return np.maximum(float(data.failure_time_s) - window_set.endpoint_times_s, 0.0)


def _current_rate_prediction(window_set: WindowSet) -> np.ndarray:
    return np.maximum(np.asarray(window_set.physics_rul_s, dtype=np.float64), 0.0)


def _local_slope_prediction(data: NozzleData, window_set: WindowSet) -> np.ndarray:
    predictions: list[float] = []
    partition_start = int(window_set.raw_partition_indices[0])
    for raw_indices, endpoint, margin, observed_rate in zip(
        window_set.raw_window_indices,
        window_set.endpoint_indices,
        window_set.depth_margin_mm,
        window_set.current_rate_m_s,
    ):
        indices = np.asarray(raw_indices, dtype=np.int64)
        # Window size one has no within-window slope.  Use one previous row only
        # when it lies in the same partition; never cross a partition boundary.
        if len(indices) < 2 and int(endpoint) > partition_start:
            indices = np.asarray([int(endpoint) - 1, int(endpoint)], dtype=np.int64)
        times = np.asarray(data.time_s[indices], dtype=np.float64)
        depths = np.asarray(data.depth_mm[indices], dtype=np.float64)
        if len(indices) >= 2 and float(np.ptp(times)) > 0.0:
            centered_time = times - float(times.mean())
            denominator = float(np.dot(centered_time, centered_time))
            slope_mm_s = float(np.dot(centered_time, depths - float(depths.mean())) / denominator)
        else:
            slope_mm_s = float(observed_rate) * 1000.0
        safe_rate_mm_s = max(slope_mm_s, float(observed_rate) * 1000.0 * 1.0e-3, 1.0e-9)
        predictions.append(max(float(margin), 0.0) / safe_rate_mm_s)
    return np.asarray(predictions, dtype=np.float64)


def _design_matrix(window_set: WindowSet) -> np.ndarray:
    values = np.asarray(window_set.X, dtype=np.float64).reshape(len(window_set), -1)
    return np.column_stack([np.ones(len(values), dtype=np.float64), values])


def _solve_weighted_linear(
    design: np.ndarray,
    target: np.ndarray,
    alpha: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weights is None:
        sample_weights = np.ones(len(target), dtype=np.float64)
    safe_weights = np.maximum(np.asarray(sample_weights, dtype=np.float64), 1.0e-12)
    root_weights = np.sqrt(safe_weights)
    weighted_design = design * root_weights[:, None]
    weighted_target = target * root_weights
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _fit_ridge(train_set: WindowSet, alpha: float) -> np.ndarray:
    return _solve_weighted_linear(
        _design_matrix(train_set),
        np.asarray(train_set.y_rul_s, dtype=np.float64),
        alpha,
    )


def _fit_huber(train_set: WindowSet, alpha: float) -> np.ndarray:
    design = _design_matrix(train_set)
    target = np.asarray(train_set.y_rul_s, dtype=np.float64)
    coefficients = _solve_weighted_linear(design, target, alpha)
    for _ in range(50):
        residual = target - design @ coefficients
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        robust_scale = max(1.4826 * mad, 1.0e-3)
        threshold = 1.345 * robust_scale
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        large = absolute > threshold
        weights[large] = threshold / np.maximum(absolute[large], 1.0e-12)
        updated = _solve_weighted_linear(design, target, alpha, weights)
        if np.linalg.norm(updated - coefficients) <= 1.0e-9 * (1.0 + np.linalg.norm(coefficients)):
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def fit_linear_baseline(
    name: str,
    train_set: WindowSet,
    val_set: WindowSet,
    failure_time_s: float,
) -> tuple[Callable[[WindowSet], np.ndarray], float, dict[str, float], float]:
    """Fit on train and select alpha on validation; there is no test argument."""

    if name not in ("ridge", "huber"):
        raise ValueError(f"unknown fitted baseline {name!r}")
    fit_function = _fit_ridge if name == "ridge" else _fit_huber
    started = time.perf_counter()
    choices: list[tuple[float, dict[str, float], np.ndarray]] = []
    val_design = _design_matrix(val_set)
    for alpha in LINEAR_ALPHAS:
        coefficients = fit_function(train_set, alpha)
        val_prediction = np.maximum(val_design @ coefficients, 0.0)
        val_metrics = compute_metrics(
            val_set.y_rul_s, val_prediction, val_set.weights, failure_time_s
        )
        choices.append((alpha, val_metrics, coefficients))
    selected_alpha, selected_metrics, selected_coefficients = min(
        choices, key=lambda item: (item[1]["rmse"], item[1]["weighted_rmse"], item[0])
    )
    training_seconds = time.perf_counter() - started

    def predictor(window_set: WindowSet) -> np.ndarray:
        return np.maximum(_design_matrix(window_set) @ selected_coefficients, 0.0)

    return predictor, float(selected_alpha), selected_metrics, training_seconds


def _baseline_run(
    name: str,
    data: NozzleData,
    prepared: PreparedFold,
) -> BaselineRun:
    if name == "clock_oracle":
        predictor = lambda window_set: _clock_prediction(data, window_set)
        selected_alpha = None
        training_seconds = 0.0
    elif name == "current_rate":
        predictor = _current_rate_prediction
        selected_alpha = None
        training_seconds = 0.0
    elif name == "local_slope":
        predictor = lambda window_set: _local_slope_prediction(data, window_set)
        selected_alpha = None
        training_seconds = 0.0
    else:
        predictor, selected_alpha, val_metrics, training_seconds = fit_linear_baseline(
            name, prepared.train, prepared.val, data.failure_time_s
        )

    if name not in ("ridge", "huber"):
        val_prediction = predictor(prepared.val)
        val_metrics = compute_metrics(
            prepared.val.y_rul_s,
            val_prediction,
            prepared.val.weights,
            data.failure_time_s,
        )
    started = time.perf_counter()
    test_prediction = predictor(prepared.test)
    inference_seconds = time.perf_counter() - started
    metrics = compute_metrics(
        prepared.test.y_rul_s,
        test_prediction,
        prepared.test.weights,
        data.failure_time_s,
    )
    return BaselineRun(
        model=name,
        fold=prepared.split.name,
        diagnostic=name in DIAGNOSTIC_METHODS,
        ranking_eligible=name in ELIGIBLE_BASELINES,
        metrics=metrics,
        val_metrics=val_metrics,
        selected_alpha=selected_alpha,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        endpoint_indices=np.asarray(prepared.test.endpoint_indices, dtype=np.int64),
        endpoint_times_s=np.asarray(prepared.test.endpoint_times_s, dtype=np.float64),
        y_true=np.asarray(prepared.test.y_rul_s, dtype=np.float64),
        y_pred=np.asarray(test_prediction, dtype=np.float64),
        weights=np.asarray(prepared.test.weights, dtype=np.float64),
        quadrature_weights_s=np.asarray(prepared.test.quadrature_weights_s, dtype=np.float64),
    )


def _prediction_rows(run: NeuralRun | BaselineRun) -> list[dict[str, Any]]:
    if isinstance(run, NeuralRun):
        common: dict[str, Any] = {
            "experiment": run.experiment,
            "method_type": "trainable",
            "model": run.model,
            "fold": run.fold,
            "seed": run.seed,
            "feature_tier": run.feature_tier,
            "window_size": run.window_size,
            "diagnostic": False,
            "ranking_eligible": True,
        }
    else:
        common = {
            "experiment": "benchmark",
            "method_type": "baseline",
            "model": run.model,
            "fold": run.fold,
            "seed": "",
            "feature_tier": PRIMARY_FEATURE_TIER,
            "window_size": PRIMARY_WINDOW_SIZE,
            "diagnostic": run.diagnostic,
            "ranking_eligible": run.ranking_eligible,
        }
    rows: list[dict[str, Any]] = []
    for index, endpoint, endpoint_time, truth, prediction, weight, quadrature in zip(
        range(len(run.endpoint_indices)),
        run.endpoint_indices,
        run.endpoint_times_s,
        run.y_true,
        run.y_pred,
        run.weights,
        run.quadrature_weights_s,
    ):
        error = float(prediction - truth)
        rows.append(
            {
                **common,
                "anchor_number": index,
                "endpoint_index": int(endpoint),
                "endpoint_time_s": float(endpoint_time),
                "y_true_rul_s": float(truth),
                "y_pred_rul_s": max(float(prediction), 0.0),
                "error_s": error,
                "absolute_error_s": abs(error),
                "squared_error_s2": error * error,
                "weight": float(weight),
                "quadrature_weight_s": float(quadrature),
            }
        )
    return rows


def _save_per_run_predictions(output_dir: Path, run: NeuralRun) -> None:
    safe_name = (
        f"{run.experiment}__{run.feature_tier}__w{run.window_size}__"
        f"{run.fold}__{run.model}__seed{run.seed}"
    )
    rows = _prediction_rows(run)
    predictions_dir = output_dir / "per_seed_predictions"
    _write_csv(predictions_dir / f"{safe_name}.csv", rows)
    _write_json(
        predictions_dir / f"{safe_name}.json",
        {"run": run.summary(), "anchors": rows},
    )


def _aggregate_neural_group(runs: Sequence[NeuralRun], failure_time_s: float) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot aggregate an empty neural run group")
    reference = runs[0]
    for run in runs[1:]:
        if not np.array_equal(run.endpoint_indices, reference.endpoint_indices):
            raise RuntimeError("seed predictions have mismatched endpoint indices")
        if not np.allclose(run.y_true, reference.y_true, rtol=0.0, atol=0.0):
            raise RuntimeError("seed predictions have mismatched labels")
    single_seed: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        values = np.asarray([run.metrics[metric] for run in runs], dtype=np.float64)
        single_seed[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    ensemble_prediction = np.mean(np.stack([run.y_pred for run in runs], axis=0), axis=0)
    ensemble_metrics = compute_metrics(
        reference.y_true, ensemble_prediction, reference.weights, failure_time_s
    )
    return {
        "model": reference.model,
        "fold": reference.fold,
        "feature_tier": reference.feature_tier,
        "window_size": reference.window_size,
        "seeds": [run.seed for run in runs],
        "n_seeds": len(runs),
        "single_seed": single_seed,
        "ensemble": {
            "rule": "fixed_equal_prediction_average",
            "weights": [1.0 / len(runs)] * len(runs),
            "metrics": ensemble_metrics,
        },
        "parameter_count": reference.parameter_count,
        "training_seconds": {
            "mean": float(np.mean([run.training_seconds for run in runs])),
            "std": float(np.std([run.training_seconds for run in runs], ddof=0)),
            "total": float(np.sum([run.training_seconds for run in runs])),
        },
        "inference_seconds": {
            "mean": float(np.mean([run.inference_seconds for run in runs])),
            "std": float(np.std([run.inference_seconds for run in runs], ddof=0)),
            "total": float(np.sum([run.inference_seconds for run in runs])),
        },
        "epochs_ran": {
            "mean": float(np.mean([run.epochs_ran for run in runs])),
            "std": float(np.std([run.epochs_ran for run in runs], ddof=0)),
        },
        "val_best_rmse": {
            "mean": float(np.mean([run.val_best["rmse"] for run in runs])),
            "std": float(np.std([run.val_best["rmse"] for run in runs], ddof=0)),
        },
        "ensemble_prediction": ensemble_prediction,
        "endpoint_indices": reference.endpoint_indices,
        "endpoint_times_s": reference.endpoint_times_s,
        "y_true": reference.y_true,
        "weights": reference.weights,
    }


def _aggregate_baseline(run: BaselineRun) -> dict[str, Any]:
    return {
        "model": run.model,
        "fold": run.fold,
        "feature_tier": PRIMARY_FEATURE_TIER,
        "window_size": PRIMARY_WINDOW_SIZE,
        "seeds": [],
        "n_seeds": 1,
        "diagnostic": run.diagnostic,
        "ranking_eligible": run.ranking_eligible,
        "single_seed": {
            metric: {"mean": run.metrics[metric], "std": 0.0} for metric in METRIC_NAMES
        },
        "ensemble": {
            "rule": "deterministic",
            "weights": [1.0],
            "metrics": run.metrics,
        },
        "parameter_count": 0,
        "training_seconds": {"mean": run.training_seconds, "std": 0.0, "total": run.training_seconds},
        "inference_seconds": {
            "mean": run.inference_seconds,
            "std": 0.0,
            "total": run.inference_seconds,
        },
        "epochs_ran": {"mean": 0.0, "std": 0.0},
        "val_best_rmse": {"mean": run.val_metrics["rmse"], "std": 0.0},
        "ensemble_prediction": run.y_pred,
        "endpoint_indices": run.endpoint_indices,
        "endpoint_times_s": run.endpoint_times_s,
        "y_true": run.y_true,
        "weights": run.weights,
    }


def _public_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    private = {"ensemble_prediction", "endpoint_indices", "endpoint_times_s", "y_true", "weights"}
    return {key: _jsonable(value) for key, value in aggregate.items() if key not in private}


def _macro_summary(fold_aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not fold_aggregates:
        raise ValueError("cannot compute a macro summary without folds")
    result: dict[str, Any] = {"folds": [item["fold"] for item in fold_aggregates]}
    for metric in METRIC_NAMES:
        ensemble_values = np.asarray(
            [item["ensemble"]["metrics"][metric] for item in fold_aggregates], dtype=np.float64
        )
        single_values = np.asarray(
            [item["single_seed"][metric]["mean"] for item in fold_aggregates], dtype=np.float64
        )
        result[f"ensemble_{metric}"] = {
            "mean": float(ensemble_values.mean()),
            "std": float(ensemble_values.std(ddof=0)),
        }
        result[f"single_seed_{metric}"] = {
            "mean": float(single_values.mean()),
            "std_across_fold_means": float(single_values.std(ddof=0)),
        }
    return result


def _benchmark_csv_rows(
    model_folds: Sequence[Mapping[str, Any]],
    baseline_folds: Sequence[Mapping[str, Any]],
    model_macros: Mapping[str, Any],
    baseline_macros: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_type, aggregates in (("trainable", model_folds), ("baseline", baseline_folds)):
        for item in aggregates:
            row: dict[str, Any] = {
                "scope": "fold",
                "method_type": method_type,
                "model": item["model"],
                "fold": item["fold"],
                "diagnostic": bool(item.get("diagnostic", False)),
                "ranking_eligible": bool(item.get("ranking_eligible", method_type == "trainable")),
                "n_seeds": item["n_seeds"],
                "parameter_count": item["parameter_count"],
                "training_seconds_mean": item["training_seconds"]["mean"],
                "inference_seconds_mean": item["inference_seconds"]["mean"],
                "epochs_mean": item["epochs_ran"]["mean"],
                "val_best_rmse_mean": item["val_best_rmse"]["mean"],
            }
            for metric in METRIC_NAMES:
                row[f"single_seed_{metric}_mean"] = item["single_seed"][metric]["mean"]
                row[f"single_seed_{metric}_std"] = item["single_seed"][metric]["std"]
                row[f"ensemble_{metric}"] = item["ensemble"]["metrics"][metric]
            rows.append(row)
    for method_type, macros in (("trainable", model_macros), ("baseline", baseline_macros)):
        for model, macro in macros.items():
            row = {
                "scope": "rolling_fold_macro",
                "method_type": method_type,
                "model": model,
                "fold": "macro",
                "diagnostic": model in DIAGNOSTIC_METHODS,
                "ranking_eligible": model not in DIAGNOSTIC_METHODS,
                "n_seeds": "",
            }
            for metric in METRIC_NAMES:
                row[f"ensemble_{metric}"] = macro[f"ensemble_{metric}"]["mean"]
                row[f"single_seed_{metric}_mean"] = macro[f"single_seed_{metric}"]["mean"]
            rows.append(row)
    return rows


def _ablation_csv_rows(ablations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ablations:
        aggregate = item["aggregate"]
        row: dict[str, Any] = {
            "scope": "fold",
            "dimension": item["dimension"],
            "variant": item["variant"],
            "model": item["model"],
            "feature_tier": item["feature_tier"],
            "window_size": item["window_size"],
            "fold": aggregate["fold"],
            "diagnostic": item["diagnostic"],
            "n_seeds": aggregate["n_seeds"],
        }
        for metric in METRIC_NAMES:
            row[f"single_seed_{metric}_mean"] = aggregate["single_seed"][metric]["mean"]
            row[f"single_seed_{metric}_std"] = aggregate["single_seed"][metric]["std"]
            row[f"ensemble_{metric}"] = aggregate["ensemble"]["metrics"][metric]
        rows.append(row)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in ablations:
        grouped.setdefault((item["dimension"], item["variant"]), []).append(item)
    for (dimension, variant), items in grouped.items():
        macro = _macro_summary([item["aggregate"] for item in items])
        first = items[0]
        row = {
            "scope": "rolling_fold_macro",
            "dimension": dimension,
            "variant": variant,
            "model": first["model"],
            "feature_tier": first["feature_tier"],
            "window_size": first["window_size"],
            "fold": "macro",
            "diagnostic": first["diagnostic"],
            "n_seeds": "",
        }
        for metric in METRIC_NAMES:
            row[f"single_seed_{metric}_mean"] = macro[f"single_seed_{metric}"]["mean"]
            row[f"ensemble_{metric}"] = macro[f"ensemble_{metric}"]["mean"]
        rows.append(row)
    return rows


def _environment(args: argparse.Namespace, device: torch.device, warning: str | None) -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created_utc": _utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": cuda_devices,
        "requested_device": args.device,
        "resolved_device": str(device),
        "device_warning": warning,
        "deterministic_algorithms": True,
        "quick": bool(args.quick),
        "phase": args.phase,
        "seeds": list(args.seeds),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _hash_file(Path(__file__).resolve()),
    }


def _frozen_payload(
    data: NozzleData,
    seeds: Sequence[int],
    quick: bool,
    validation_results: Sequence[Mapping[str, Any]],
    selected_config: Mapping[str, Any],
) -> dict[str, Any]:
    split_specs = {name: spec.as_dict() for name, spec in get_split_specs().items()}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "frozen_before_test_evaluation",
        "quick": bool(quick),
        "quick_disclaimer": (
            "reduced seeds/epochs; smoke-test output is not the canonical five-seed benchmark"
            if quick
            else None
        ),
        "requested_seeds": list(seeds),
        "effective_final_seeds": [int(seeds[0])] if quick else list(seeds),
        "selection_seed": int(seeds[0]),
        "protocol": PROTOCOL_SPEC,
        "split_specs": split_specs,
        "hashes": {
            "dataset_sha256": data.manifest.sha256,
            "expected_dataset_sha256": CANONICAL_SHA256,
            "script_sha256": _hash_file(Path(__file__).resolve()),
            "protocol_sha256": _hash_json(PROTOCOL_SPEC),
            "candidates_sha256": _hash_json(CANDIDATES),
            "split_specs_sha256": _hash_json(split_specs),
        },
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "validation_metrics": list(validation_results),
        "selected_config": dict(selected_config),
        "selection_rule_applied": PROTOCOL_SPEC["selection"]["objective"],
    }
    payload["config_hash"] = _hash_json(payload)
    return payload


def run_select(
    data_path: Path,
    output_dir: Path,
    device: torch.device,
    seeds: Sequence[int],
    quick: bool,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Select a candidate without reading any PreparedFold.test attribute."""

    data = load_nozzle_csv(data_path)
    validation_results: list[dict[str, Any]] = []
    selection_seed = int(seeds[0])
    print(
        f"SELECT: {len(CANDIDATES)} candidates on {SELECTION_FOLDS}, seed={selection_seed}",
        flush=True,
    )
    for candidate in CANDIDATES:
        fold_rows: list[dict[str, Any]] = []
        for fold_name in SELECTION_FOLDS:
            prepared = prepare_fold(
                data,
                fold_name,
                feature_tier=PRIMARY_FEATURE_TIER,
                window_size=PRIMARY_WINDOW_SIZE,
            )
            # Only these two partitions enter the trainer.  Do not bind or read
            # prepared.test anywhere in this selection phase.
            model, info = train_model(
                PRIMARY_MODEL,
                n_features=len(prepared.feature_names),
                train_set=prepared.train,
                val_set=prepared.val,
                config=candidate,
                device=device,
                seed=selection_seed,
                failure_time_s=data.failure_time_s,
                quick=quick,
            )
            del model
            fold_rows.append(
                {
                    "fold": fold_name,
                    "validation_metrics": info["best_val_metrics"],
                    "best_epoch": info["best_epoch"],
                    "epochs_ran": info["epochs_ran"],
                    "training_seconds": info["training_seconds"],
                    "parameter_count": info["parameter_count"],
                    "effective_config": info["effective_config"],
                }
            )
        macro_rmse = float(np.mean([row["validation_metrics"]["rmse"] for row in fold_rows]))
        macro_weighted = float(
            np.mean([row["validation_metrics"]["weighted_rmse"] for row in fold_rows])
        )
        validation_results.append(
            {
                "candidate_id": candidate["id"],
                "model": PRIMARY_MODEL,
                "seed": selection_seed,
                "folds": fold_rows,
                "macro_validation_rmse": macro_rmse,
                "macro_validation_weighted_rmse": macro_weighted,
            }
        )
        print(
            f"  {candidate['id']}: val macro RMSE={macro_rmse:.6f} s",
            flush=True,
        )
    winner = min(
        validation_results,
        key=lambda row: (
            row["macro_validation_rmse"],
            row["macro_validation_weighted_rmse"],
            row["candidate_id"],
        ),
    )
    selected_config = next(
        candidate for candidate in CANDIDATES if candidate["id"] == winner["candidate_id"]
    )
    frozen = _frozen_payload(data, seeds, quick, validation_results, selected_config)
    _write_json(output_dir / "frozen_config.json", frozen)
    selection_environment = dict(environment)
    selection_environment.update(
        {
            "dataset_manifest": data.manifest.as_dict(),
            "config_hash": frozen["config_hash"],
        }
    )
    _write_json(output_dir / "environment.json", selection_environment)
    print(
        f"  selected={selected_config['id']} config_hash={frozen['config_hash']}", flush=True
    )
    return frozen


def _validate_frozen(
    frozen: Mapping[str, Any],
    data: NozzleData,
    seeds: Sequence[int],
    quick: bool,
) -> None:
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("frozen_config.json schema_version does not match this script")
    supplied_hash = frozen.get("config_hash")
    if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
        raise ValueError("frozen_config.json has no valid config_hash")
    unhashed = dict(frozen)
    unhashed.pop("config_hash", None)
    actual_hash = _hash_json(unhashed)
    if actual_hash != supplied_hash:
        raise ValueError(
            f"frozen_config.json config_hash mismatch: expected {supplied_hash}, computed {actual_hash}"
        )
    hashes = frozen.get("hashes", {})
    expected = {
        "dataset_sha256": data.manifest.sha256,
        "expected_dataset_sha256": CANONICAL_SHA256,
        "script_sha256": _hash_file(Path(__file__).resolve()),
        "protocol_sha256": _hash_json(PROTOCOL_SPEC),
        "candidates_sha256": _hash_json(CANDIDATES),
        "split_specs_sha256": _hash_json(
            {name: spec.as_dict() for name, spec in get_split_specs().items()}
        ),
    }
    for key, expected_value in expected.items():
        if hashes.get(key) != expected_value:
            raise ValueError(
                f"frozen_config.json {key} mismatch: frozen={hashes.get(key)!r}, current={expected_value!r}"
            )
    if list(frozen.get("requested_seeds", [])) != list(seeds):
        raise ValueError("--seeds must exactly match requested_seeds in frozen_config.json")
    if bool(frozen.get("quick")) != bool(quick):
        raise ValueError("--quick setting must match frozen_config.json")
    if frozen.get("protocol") != PROTOCOL_SPEC:
        raise ValueError("frozen protocol payload differs from the current protocol")
    if frozen.get("candidates") != [dict(candidate) for candidate in CANDIDATES]:
        raise ValueError("frozen candidate set differs from the predeclared candidate set")
    selected = frozen.get("selected_config")
    if selected not in [dict(candidate) for candidate in CANDIDATES]:
        raise ValueError("selected_config is not one of the predeclared candidates")


def _execute_neural_run(
    experiment: str,
    model_name: str,
    prepared: PreparedFold,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    quick: bool,
    output_dir: Path,
) -> NeuralRun:
    model, info = train_model(
        model_name,
        n_features=len(prepared.feature_names),
        train_set=prepared.train,
        val_set=prepared.val,
        config=config,
        device=device,
        seed=seed,
        failure_time_s=prepared.manifest.failure_time_s,
        quick=quick,
    )
    test_prediction, inference_seconds = predict_partition(
        model, model_name, prepared.test, device
    )
    metrics = compute_metrics(
        prepared.test.y_rul_s,
        test_prediction,
        prepared.test.weights,
        prepared.manifest.failure_time_s,
    )
    run = NeuralRun(
        experiment=experiment,
        model=model_name,
        fold=prepared.split.name,
        seed=seed,
        feature_tier=prepared.feature_tier,
        window_size=prepared.window_size,
        metrics=metrics,
        val_best=info["best_val_metrics"],
        best_epoch=info["best_epoch"],
        epochs_ran=info["epochs_ran"],
        parameter_count=info["parameter_count"],
        training_seconds=info["training_seconds"],
        inference_seconds=inference_seconds,
        loss_scales=info["loss_scales"],
        endpoint_indices=np.asarray(prepared.test.endpoint_indices, dtype=np.int64),
        endpoint_times_s=np.asarray(prepared.test.endpoint_times_s, dtype=np.float64),
        y_true=np.asarray(prepared.test.y_rul_s, dtype=np.float64),
        y_pred=test_prediction,
        weights=np.asarray(prepared.test.weights, dtype=np.float64),
        quadrature_weights_s=np.asarray(
            prepared.test.quadrature_weights_s, dtype=np.float64
        ),
    )
    _save_per_run_predictions(output_dir, run)
    return run


def _claim_audit(
    neural_runs: Sequence[NeuralRun],
    model_macros: Mapping[str, Any],
    baseline_macros: Mapping[str, Any],
    model_folds: Sequence[Mapping[str, Any]],
    baseline_folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary_macro_rmse = model_macros[PRIMARY_MODEL]["ensemble_rmse"]["mean"]
    baseline_macro_rmse = {
        name: baseline_macros[name]["ensemble_rmse"]["mean"] for name in ELIGIBLE_BASELINES
    }
    locked_primary = next(
        item for item in model_folds if item["model"] == PRIMARY_MODEL and item["fold"] == LOCKED_FOLD
    )
    locked_baselines = {
        item["model"]: item
        for item in baseline_folds
        if item["fold"] == LOCKED_FOLD and item["model"] in ELIGIBLE_BASELINES
    }
    locked_beats_all = all(
        locked_primary["ensemble"]["metrics"]["rmse"]
        < locked_baselines[name]["ensemble"]["metrics"]["rmse"]
        for name in ELIGIBLE_BASELINES
    )
    rolling_beats_all = all(
        primary_macro_rmse < baseline_macro_rmse[name] for name in ELIGIBLE_BASELINES
    )
    baseline_by_fold = {
        (item["fold"], item["model"]): item["ensemble"]["metrics"]["rmse"]
        for item in baseline_folds
        if item["model"] in ELIGIBLE_BASELINES
    }
    primary_cells = [
        run for run in neural_runs if run.experiment == "benchmark" and run.model == PRIMARY_MODEL
    ]
    cell_results = [
        {
            "fold": run.fold,
            "seed": run.seed,
            "primary_rmse": run.metrics["rmse"],
            "beats_all_eligible_baselines": all(
                run.metrics["rmse"] < baseline_by_fold[(run.fold, name)]
                for name in ELIGIBLE_BASELINES
            ),
        }
        for run in primary_cells
    ]
    cells_won = sum(bool(row["beats_all_eligible_baselines"]) for row in cell_results)
    majority_required = len(cell_results) // 2 + 1
    majority_won = cells_won >= majority_required
    supports_maximum = locked_beats_all and rolling_beats_all and majority_won

    all_ranked_macro = {
        **{name: model_macros[name]["ensemble_rmse"]["mean"] for name in MODEL_FAMILIES},
        **baseline_macro_rmse,
    }
    ranked_macro_for_current_rate = {
        name: value for name, value in all_ranked_macro.items() if name not in DIAGNOSTIC_METHODS
    }
    current_rate_wins = ranked_macro_for_current_rate["current_rate"] <= min(
        ranked_macro_for_current_rate.values()
    )
    if supports_maximum:
        automatic_result = "task-specific strict best on this single trajectory"
    elif current_rate_wins:
        automatic_result = (
            "The deterministic current-rate baseline wins the eligible rolling-fold macro RMSE comparison."
        )
    else:
        automatic_result = (
            "The preregistered claim criteria are not all satisfied; no task-specific best claim is made."
        )
    return {
        "automatic_result": automatic_result,
        "maximum_permitted_claim": PROTOCOL_SPEC["claim_rule"]["maximum_text"],
        "global_sota_claim": False,
        "primary_model": PRIMARY_MODEL,
        "eligible_baselines": list(ELIGIBLE_BASELINES),
        "diagnostic_exclusions": ["clock_oracle", "age_aware ablation"],
        "criteria": {
            "locked_fold_rmse_beats_all": locked_beats_all,
            "rolling_fold_macro_rmse_beats_all": rolling_beats_all,
            "majority_seed_fold_cells": majority_won,
        },
        "majority_details": {
            "cells_won": cells_won,
            "cells_total": len(cell_results),
            "cells_required": majority_required,
            "cells": cell_results,
        },
        "current_rate_wins_ranked_macro": current_rate_wins,
        "rolling_macro_rmse": all_ranked_macro,
    }


def _make_plots(
    output_dir: Path,
    model_macros: Mapping[str, Any],
    baseline_macros: Mapping[str, Any],
    model_folds: Sequence[Mapping[str, Any]],
    baseline_folds: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    artifacts: list[str] = []
    warnings: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return artifacts, [f"matplotlib unavailable; plots skipped: {exc}"]
    try:
        labels = list(MODEL_FAMILIES) + list(ELIGIBLE_BASELINES)
        values = [model_macros[name]["ensemble_rmse"]["mean"] for name in MODEL_FAMILIES]
        values += [baseline_macros[name]["ensemble_rmse"]["mean"] for name in ELIGIBLE_BASELINES]
        figure, axis = plt.subplots(figsize=(10, 4.8))
        axis.bar(range(len(labels)), values, color=["#4C78A8"] * 5 + ["#F58518"] * 4)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_ylabel("Rolling-fold macro RMSE (s)")
        axis.set_title("Strict nozzle benchmark (equal-seed ensembles)")
        figure.tight_layout()
        path = output_dir / "benchmark_rmse.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        artifacts.append(path.name)

        primary = next(
            item for item in model_folds if item["model"] == PRIMARY_MODEL and item["fold"] == LOCKED_FOLD
        )
        current = next(
            item for item in baseline_folds if item["model"] == "current_rate" and item["fold"] == LOCKED_FOLD
        )
        local = next(
            item for item in baseline_folds if item["model"] == "local_slope" and item["fold"] == LOCKED_FOLD
        )
        figure, axis = plt.subplots(figsize=(7, 4.8))
        axis.plot(primary["endpoint_times_s"], primary["y_true"], "ko-", label="true RUL")
        axis.plot(
            primary["endpoint_times_s"],
            primary["ensemble_prediction"],
            "o-",
            label=f"{PRIMARY_MODEL} equal-seed ensemble",
        )
        axis.plot(current["endpoint_times_s"], current["ensemble_prediction"], "s--", label="current_rate")
        axis.plot(local["endpoint_times_s"], local["ensemble_prediction"], "^--", label="local_slope")
        axis.set_xlabel("Anchor time (s)")
        axis.set_ylabel("RUL (s)")
        axis.set_title("Locked fold3 predictions")
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = output_dir / "locked_fold3_predictions.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        artifacts.append(path.name)

        macro_rows = [row for row in ablation_rows if row["scope"] == "rolling_fold_macro"]
        labels = [f"{row['dimension']}:{row['variant']}" for row in macro_rows]
        values = [float(row["ensemble_rmse"]) for row in macro_rows]
        figure, axis = plt.subplots(figsize=(11, 5.5))
        axis.bar(range(len(labels)), values, color="#54A24B")
        axis.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
        axis.set_ylabel("Rolling-fold macro RMSE (s)")
        axis.set_title("Strict nozzle ablations")
        figure.tight_layout()
        path = output_dir / "ablation_rmse.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        artifacts.append(path.name)
    except Exception as exc:  # Plot failure must not replace or invalidate numeric results.
        warnings.append(f"plot generation failed after numeric results completed: {exc}")
    return artifacts, warnings


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict Nozzle RUL Benchmark",
        "",
        f"- Created: `{report['created_utc']}`",
        f"- Frozen config hash: `{report['frozen_config']['config_hash']}`",
        f"- Dataset SHA256: `{report['dataset']['sha256']}`",
        f"- Execution mode: `{'quick smoke test' if report['run']['quick'] else 'canonical'}`",
        f"- Automatic claim audit: **{report['claim_audit']['automatic_result']}**",
        "- Test policy: validation-only early stopping; test predicted once per locked run; no test-fitted seed, family, or ensemble weights.",
        "",
    ]
    if report["run"]["quick"]:
        lines.extend(
            [
                "> This is a reduced quick smoke test, not the canonical five-seed benchmark.",
                "",
            ]
        )
    lines.extend(
        [
            "## Rolling-fold macro metrics",
            "",
            "| Method | Type | RMSE (s) | MAE (s) | Weighted RMSE (s) | NRMSE | Ranking |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    model_macros = report["benchmark"]["rolling_fold_macro"]["trainable"]
    baseline_macros = report["benchmark"]["rolling_fold_macro"]["baselines"]
    for model in MODEL_FAMILIES:
        macro = model_macros[model]
        lines.append(
            f"| {model} | trainable | {macro['ensemble_rmse']['mean']:.6f} | "
            f"{macro['ensemble_mae']['mean']:.6f} | {macro['ensemble_weighted_rmse']['mean']:.6f} | "
            f"{macro['ensemble_nrmse']['mean']:.6f} | eligible |"
        )
    for model in ("current_rate", "local_slope", "ridge", "huber", "clock_oracle"):
        macro = baseline_macros[model]
        ranking = "diagnostic/excluded" if model == "clock_oracle" else "eligible"
        lines.append(
            f"| {model} | baseline | {macro['ensemble_rmse']['mean']:.6f} | "
            f"{macro['ensemble_mae']['mean']:.6f} | {macro['ensemble_weighted_rmse']['mean']:.6f} | "
            f"{macro['ensemble_nrmse']['mean']:.6f} | {ranking} |"
        )
    lines.extend(
        [
            "",
            "## Locked fold3 metrics",
            "",
            "| Method | RMSE (s) | MAE (s) | Weighted RMSE (s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["benchmark"]["locked_fold3"]:
        metrics = item["ensemble"]["metrics"]
        lines.append(
            f"| {item['model']} | {metrics['rmse']:.6f} | {metrics['mae']:.6f} | "
            f"{metrics['weighted_rmse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Ablations (rolling-fold macro)",
            "",
            "| Dimension | Variant | Model | Window | RMSE (s) | Status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in report["ablations"]["summary_rows"]:
        if row["scope"] != "rolling_fold_macro":
            continue
        status = "diagnostic/excluded" if row["diagnostic"] else "predeclared"
        lines.append(
            f"| {row['dimension']} | {row['variant']} | {row['model']} | "
            f"{row['window_size']} | {float(row['ensemble_rmse']):.6f} | {status} |"
        )
    lines.extend(
        [
            "",
            "The feature/window ablations use the predeclared TCN control so the external physics-RUL branch does not bypass a feature tier. The physics ablation is TCN vs physics-input vs full-physics at state-aware/window-3.",
            "",
            "## Claim boundary",
            "",
            report["claim_audit"]["automatic_result"],
            "This single simulated trajectory cannot support a global SOTA claim or external-generalization claim.",
            "Age-aware inputs and the exact `20-t` clock oracle are diagnostic and excluded from ranking.",
            "",
            "## External same-domain status",
            "",
            "**N/A.** No verified machine-readable public same-task nozzle RUL dataset was identified. See `docs/nozzle_public_data_audit.md`.",
            "",
            "## Historical non-strict numbers (separate; not comparable)",
            "",
            "Legacy values `2.501 / 2.674 / 3.819` are retained only as historical non-strict context. They are not substituted for any failed run and are not included in strict rankings or claims.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for artifact in report["artifacts"]:
        lines.append(f"- `{artifact}`")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def run_final(
    data_path: Path,
    output_dir: Path,
    device: torch.device,
    seeds: Sequence[int],
    quick: bool,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_path = output_dir / "frozen_config.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(
            f"FINAL requires {frozen_path}; run --phase select first (or use --phase all)"
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    data = load_nozzle_csv(data_path)
    _validate_frozen(frozen, data, seeds, quick)
    selected_config = frozen["selected_config"]
    effective_seeds = tuple(int(seed) for seed in frozen["effective_final_seeds"])
    started = time.perf_counter()
    print(
        f"FINAL: config={selected_config['id']} folds={len(FINAL_FOLDS)} "
        f"seeds={effective_seeds} device={device}",
        flush=True,
    )

    prepared_cache: dict[tuple[str, str, int], PreparedFold] = {}

    def prepared(fold: str, tier: str, window: int) -> PreparedFold:
        key = (fold, tier, window)
        if key not in prepared_cache:
            prepared_cache[key] = prepare_fold(data, fold, feature_tier=tier, window_size=window)
        return prepared_cache[key]

    neural_runs: list[NeuralRun] = []
    baseline_runs: list[BaselineRun] = []
    for fold_name in FINAL_FOLDS:
        primary_fold = prepared(fold_name, PRIMARY_FEATURE_TIER, PRIMARY_WINDOW_SIZE)
        for baseline_name in ("current_rate", "local_slope", "ridge", "huber", "clock_oracle"):
            baseline_runs.append(_baseline_run(baseline_name, data, primary_fold))
        for model_name in MODEL_FAMILIES:
            for seed in effective_seeds:
                run = _execute_neural_run(
                    "benchmark",
                    model_name,
                    primary_fold,
                    selected_config,
                    device,
                    seed,
                    quick,
                    output_dir,
                )
                neural_runs.append(run)
                print(
                    f"  benchmark {fold_name}/{model_name}/seed{seed}: "
                    f"RMSE={run.metrics['rmse']:.5f} val={run.val_best['rmse']:.5f}",
                    flush=True,
                )

    # Feature and window ablations use the predeclared non-physics TCN control;
    # state-aware/window-3 was already executed above and is reused exactly.
    extra_combinations = [
        (tier, PRIMARY_WINDOW_SIZE)
        for tier in FEATURE_TIERS
        if tier != PRIMARY_FEATURE_TIER
    ] + [
        (PRIMARY_FEATURE_TIER, window) for window in (1, 5)
    ]
    for tier, window in extra_combinations:
        for fold_name in FINAL_FOLDS:
            fold_data = prepared(fold_name, tier, window)
            for seed in effective_seeds:
                run = _execute_neural_run(
                    "ablation",
                    FEATURE_WINDOW_ABLATION_MODEL,
                    fold_data,
                    selected_config,
                    device,
                    seed,
                    quick,
                    output_dir,
                )
                neural_runs.append(run)
                print(
                    f"  ablation {tier}/w{window}/{fold_name}/seed{seed}: "
                    f"RMSE={run.metrics['rmse']:.5f}",
                    flush=True,
                )

    # Aggregate benchmark models by fold without selecting any family or seed.
    model_fold_aggregates: list[dict[str, Any]] = []
    for model_name in MODEL_FAMILIES:
        for fold_name in FINAL_FOLDS:
            group = [
                run
                for run in neural_runs
                if run.experiment == "benchmark"
                and run.model == model_name
                and run.fold == fold_name
            ]
            model_fold_aggregates.append(_aggregate_neural_group(group, data.failure_time_s))
    baseline_fold_aggregates = [_aggregate_baseline(run) for run in baseline_runs]
    model_macros = {
        model: _macro_summary(
            [item for item in model_fold_aggregates if item["model"] == model]
        )
        for model in MODEL_FAMILIES
    }
    baseline_macros = {
        model: _macro_summary(
            [item for item in baseline_fold_aggregates if item["model"] == model]
        )
        for model in ("current_rate", "local_slope", "ridge", "huber", "clock_oracle")
    }

    run_lookup: dict[tuple[str, str, int], list[NeuralRun]] = {}
    for run in neural_runs:
        run_lookup.setdefault((run.model, run.feature_tier, run.window_size), []).append(run)
    ablation_items: list[dict[str, Any]] = []
    ablation_definitions: list[tuple[str, str, str, str, int, bool]] = []
    for tier in FEATURE_TIERS:
        ablation_definitions.append(
            ("feature_tier", tier, FEATURE_WINDOW_ABLATION_MODEL, tier, 3, tier == "age_aware")
        )
    for window in (1, 3, 5):
        ablation_definitions.append(
            ("window_size", str(window), FEATURE_WINDOW_ABLATION_MODEL, PRIMARY_FEATURE_TIER, window, False)
        )
    for model_name in ("tcn", "physics_input", "physics_full"):
        ablation_definitions.append(
            ("physics", model_name, model_name, PRIMARY_FEATURE_TIER, 3, False)
        )
    for dimension, variant, model_name, tier, window, diagnostic in ablation_definitions:
        candidates = run_lookup[(model_name, tier, window)]
        for fold_name in FINAL_FOLDS:
            group = [run for run in candidates if run.fold == fold_name]
            aggregate = _aggregate_neural_group(group, data.failure_time_s)
            ablation_items.append(
                {
                    "dimension": dimension,
                    "variant": variant,
                    "model": model_name,
                    "feature_tier": tier,
                    "window_size": window,
                    "diagnostic": diagnostic,
                    "aggregate": aggregate,
                }
            )

    benchmark_rows = _benchmark_csv_rows(
        model_fold_aggregates, baseline_fold_aggregates, model_macros, baseline_macros
    )
    ablation_rows = _ablation_csv_rows(ablation_items)
    prediction_rows: list[dict[str, Any]] = []
    for run in neural_runs:
        prediction_rows.extend(_prediction_rows(run))
    for run in baseline_runs:
        prediction_rows.extend(_prediction_rows(run))
    _write_csv(output_dir / "benchmark_summary.csv", benchmark_rows)
    _write_csv(output_dir / "ablation_summary.csv", ablation_rows)
    _write_csv(output_dir / "predictions.csv", prediction_rows)

    claim_audit = _claim_audit(
        neural_runs,
        model_macros,
        baseline_macros,
        model_fold_aggregates,
        baseline_fold_aggregates,
    )
    plot_artifacts, plot_warnings = _make_plots(
        output_dir,
        model_macros,
        baseline_macros,
        model_fold_aggregates,
        baseline_fold_aggregates,
        ablation_rows,
    )
    elapsed = time.perf_counter() - started
    final_environment = dict(environment)
    final_environment.update(
        {
            "dataset_manifest": data.manifest.as_dict(),
            "config_hash": frozen["config_hash"],
            "effective_final_seeds": list(effective_seeds),
            "elapsed_seconds": elapsed,
        }
    )
    _write_json(output_dir / "environment.json", final_environment)

    locked_fold = [
        _public_aggregate(item)
        for item in model_fold_aggregates + baseline_fold_aggregates
        if item["fold"] == LOCKED_FOLD
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "run": {
            "phase": "final",
            "quick": quick,
            "requested_seeds": list(seeds),
            "effective_seeds": list(effective_seeds),
            "device": str(device),
            "elapsed_seconds": elapsed,
            "test_evaluation": "one stored prediction pass per final locked run",
        },
        "protocol": PROTOCOL_SPEC,
        "dataset": data.manifest.as_dict(),
        "frozen_config": {
            "path": str(frozen_path.resolve()),
            "config_hash": frozen["config_hash"],
            "selected_config": selected_config,
            "selection_validation_metrics": frozen["validation_metrics"],
            "hashes": frozen["hashes"],
        },
        "benchmark": {
            "per_seed_runs": [
                run.summary() for run in neural_runs if run.experiment == "benchmark"
            ],
            "baseline_runs": [run.summary() for run in baseline_runs],
            "fold_model_aggregates": [
                _public_aggregate(item) for item in model_fold_aggregates
            ],
            "fold_baseline_aggregates": [
                _public_aggregate(item) for item in baseline_fold_aggregates
            ],
            "rolling_fold_macro": {
                "trainable": model_macros,
                "baselines": baseline_macros,
            },
            "locked_fold3": locked_fold,
            "selection_note": "All families/seeds are reported; none is selected using test labels.",
        },
        "ablations": {
            "per_seed_runs": [
                run.summary() for run in neural_runs if run.experiment == "ablation"
            ],
            "fold_aggregates": [
                {
                    **{key: value for key, value in item.items() if key != "aggregate"},
                    "aggregate": _public_aggregate(item["aggregate"]),
                }
                for item in ablation_items
            ],
            "summary_rows": ablation_rows,
            "age_aware_ranking_status": "diagnostic/excluded",
            "test_tuning": False,
        },
        "claim_audit": claim_audit,
        "external_same_domain": {
            "status": "N/A",
            "reason": "no verified machine-readable public same-task nozzle RUL dataset",
            "audit_document": "docs/nozzle_public_data_audit.md",
        },
        "historical_non_strict": {
            "values": [2.501, 2.674, 3.819],
            "status": "separate historical context only; not comparable or ranked",
            "used_as_failure_fallback": False,
        },
        "warnings": plot_warnings,
        "artifacts": [
            "frozen_config.json",
            "NOZZLE_STRICT_REPORT.json",
            "NOZZLE_STRICT_REPORT.md",
            "benchmark_summary.csv",
            "ablation_summary.csv",
            "predictions.csv",
            "environment.json",
            "per_seed_predictions/",
            *plot_artifacts,
        ],
    }
    _write_json(output_dir / "NOZZLE_STRICT_REPORT.json", report)
    _write_text(output_dir / "NOZZLE_STRICT_REPORT.md", _markdown_report(report))
    print(f"FINAL complete in {elapsed:.2f}s", flush=True)
    print(f"  {claim_audit['automatic_result']}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage leakage-safe strict nozzle benchmark"
    )
    parser.add_argument("--phase", choices=("select", "final", "all"), default="all")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nozzle_ablation_full.csv",
        help="canonical strict nozzle CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "nozzle_strict",
        help="output directory; FINAL reads frozen_config.json here",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated fixed seeds (default: 42,123,456,2026,3407)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke-test mode: one effective seed and capped epochs; not canonical",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_runtime()
    try:
        device, device_warning = resolve_device(args.device)
        if device_warning:
            print(f"WARNING: {device_warning}", file=sys.stderr, flush=True)
        environment = _environment(args, device, device_warning)
        if args.phase in ("select", "all"):
            run_select(
                args.data,
                args.output,
                device,
                args.seeds,
                args.quick,
                environment,
            )
        if args.phase in ("final", "all"):
            run_final(
                args.data,
                args.output,
                device,
                args.seeds,
                args.quick,
                environment,
            )
        return 0
    except Exception as exc:
        failure = {
            "created_utc": _utc_now(),
            "phase": args.phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_constants_substituted": False,
        }
        try:
            _write_json(args.output / "run_failure.json", failure)
        except Exception:
            pass
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
