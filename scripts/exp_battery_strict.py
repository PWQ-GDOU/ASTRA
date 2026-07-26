"""Strict nested battery RUL benchmark (nozzle frozen).

The outer split is cell-level leave-one-cell-out.  Inner selection is also
cell-level, so no validation window shares cycles with a training window.
B0007 strict14 is right-censored and is reported separately from event-observed
strict14 cells.  The script never uses an outer test label for selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.battery_strict import (
    BATTERY_NAMES,
    FEATURE_NAMES,
    BatterySeries,
    BatteryWindowSet,
    RawBattery,
    describe_series,
    fit_scaler,
    load_battery_mat,
    make_windows,
    materialize_battery,
)
from src.models.battery_strict import RUL_SCALE, build_battery_model, count_parameters


FINAL_SEEDS = (42, 123, 456, 2026, 3407)
SELECTION_SEEDS = FINAL_SEEDS
COMMON_ENDPOINT = 23
NRMSE_CYCLES = 100.0
BLEND_ALPHA_STEP = 0.01
DEFAULT_FEATURES = FEATURE_NAMES
FIXED_FAMILY_NAMES = ("ms_l16", "life_l16", "gru_l16", "transformer_l16")


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    seq_len: int
    use_age: bool = True
    use_capacity_aux: bool = True


RIDGE_REFERENCE_CANDIDATE = Candidate("ridge_reference_l16", "ms", 16)


CANDIDATES = (
    Candidate("ms_l16", "ms", 16),
    Candidate("gru_l16", "gru", 16),
    Candidate("transformer_l16", "transformer", 16),
    Candidate("life_l8", "life", 8),
    Candidate("life_l16", "life", 16),
    Candidate("life_l24", "life", 24),
    Candidate("life_l16_no_aux", "life", 16, use_capacity_aux=False),
    Candidate("life_l16_no_age", "life", 16, use_age=False),
)


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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_pred[mask] - y_true[mask]))


def finite_mean(values: Iterable[float | None]) -> float:
    finite_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite_values.append(numeric)
    return float(np.mean(finite_values)) if finite_values else float("nan")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_dict(candidate: Candidate) -> dict[str, object]:
    return asdict(candidate)


def select_features(candidate: Candidate) -> tuple[str, ...]:
    if candidate.use_age:
        return tuple(DEFAULT_FEATURES)
    return tuple(name for name in DEFAULT_FEATURES if name != "cycle_age")


def age_values(window_set: BatteryWindowSet) -> np.ndarray:
    return window_set.endpoints.astype(np.float32)


def _cell_weights(cells: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(cells.astype(str), return_counts=True)
    lookup = {name: 1.0 / float(count) for name, count in zip(unique, counts)}
    weights = np.asarray([lookup[str(name)] for name in cells], dtype=np.float32)
    return weights / max(float(weights.mean()), 1.0e-8)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    if not bool(mask.any()):
        return None
    return values[mask].mean()


def _loss(
    output,
    batch: BatteryWindowSet,
    model,
    device: str,
) -> torch.Tensor:
    exact = torch.as_tensor(batch.target_mask, device=device)
    lower = torch.as_tensor(batch.lower_bound_rul, dtype=torch.float32, device=device)
    rul = output.rul
    per_item = torch.zeros_like(rul)
    if bool(exact.any()):
        y = torch.as_tensor(batch.rul, dtype=torch.float32, device=device)
        # Keep the objective numerically aligned with normalized features while
        # retaining cycle units at the model/evaluator boundary.
        per_item[exact] = RUL_SCALE * F.smooth_l1_loss(
            rul[exact] / RUL_SCALE,
            y[exact] / RUL_SCALE,
            beta=5.0 / RUL_SCALE,
            reduction="none",
        )
    censored = ~exact
    if bool(censored.any()):
        # At an observed endpoint of a censored trajectory, true RUL is at
        # least the remaining observed-record length.  Penalize violations only.
        per_item[censored] = RUL_SCALE * F.relu(
            (lower[censored] - rul[censored]) / RUL_SCALE
        ).square()
    weights = torch.as_tensor(_cell_weights(batch.cells), dtype=torch.float32, device=device)
    total = (per_item * weights).mean()
    if getattr(model, "use_capacity_aux", True):
        d5 = torch.as_tensor(batch.future_delta_5, dtype=torch.float32, device=device)
        d10 = torch.as_tensor(batch.future_delta_10, dtype=torch.float32, device=device)
        m5 = torch.as_tensor(batch.future_mask_5, device=device)
        m10 = torch.as_tensor(batch.future_mask_10, device=device)
        if bool(m5.any()):
            total = total + 0.15 * F.smooth_l1_loss(output.capacity_delta_5[m5], d5[m5], beta=0.02)
        if bool(m10.any()):
            total = total + 0.15 * F.smooth_l1_loss(output.capacity_delta_10[m10], d10[m10], beta=0.02)
    return total


def validation_score(batch: BatteryWindowSet, predictions: np.ndarray) -> float:
    exact = batch.target_mask
    if np.any(exact):
        return rmse(batch.rul[exact], predictions[exact])
    # A censored-only validation cell cannot yield an RMSE.  Use the mean
    # lower-bound violation as a diagnostic selection score instead.
    return float(np.mean(np.maximum(batch.lower_bound_rul - predictions, 0.0)))


def train_model(
    candidate: Candidate,
    train_set: BatteryWindowSet,
    val_set: BatteryWindowSet | None,
    device: str,
    seed: int,
    *,
    epochs: int,
    learning_rate: float = 7.0e-4,
    batch_size: int = 2048,
    patience: int = 25,
    scheduler_epoch_budget: int | None = None,
) -> tuple[object, int, float]:
    # This call must precede model construction so seed controls initialization.
    seed_everything(seed)
    model = build_battery_model(
        candidate.model,
        n_features=train_set.X.shape[-1],
        hidden=72 if candidate.model != "transformer" else 48,
        use_age=candidate.use_age,
        use_capacity_aux=candidate.use_capacity_aux,
    ).to(device)
    x = torch.as_tensor(train_set.X, dtype=torch.float32, device=device)
    age = torch.as_tensor(age_values(train_set), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    batches_per_epoch = math.ceil(max(len(train_set), 1) / batch_size)
    scheduler_budget = max(epochs, int(scheduler_epoch_budget or epochs))
    steps = max(1, scheduler_budget * batches_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = float("inf")
    best_epoch = epochs
    stale = 0
    val_x = val_age = None
    if val_set is not None:
        val_x = torch.as_tensor(val_set.X, dtype=torch.float32, device=device)
        val_age = torch.as_tensor(age_values(val_set), dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_set))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            mini = train_set.subset(idx)
            output = model(x[idx], age[idx])
            loss = _loss(output, mini, model, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        if val_set is None:
            # Final training is fixed-epoch training from the frozen config;
            # there is intentionally no checkpoint selection in this branch.
            best_epoch = epoch
            continue
        model.eval()
        with torch.no_grad():
            val_output = model(val_x, val_age)
            val_pred = val_output.rul.detach().cpu().numpy()
        score = validation_score(val_set, val_pred)
        if score < best_score - 1.0e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_metadata = {
        "trained_epoch_limit": int(epochs),
        "scheduler_epoch_budget": int(scheduler_budget),
        "batches_per_epoch": int(batches_per_epoch),
    }
    model.eval()
    return model, best_epoch, float(best_score)


@torch.no_grad()
def predict_model(model, window_set: BatteryWindowSet, device: str) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(window_set.X, dtype=torch.float32, device=device)
    age = torch.as_tensor(age_values(window_set), dtype=torch.float32, device=device)
    return model(x, age).rul.detach().cpu().numpy().astype(np.float32)


def _regression_matrix(window_set: BatteryWindowSet) -> np.ndarray:
    last = window_set.X[:, -1, :]
    age = window_set.endpoints.astype(np.float64)[:, None]
    return np.concatenate([np.ones((len(window_set), 1)), last.astype(np.float64), age / 100.0], axis=1)


def fit_ridge(train_set: BatteryWindowSet, alpha: float = 10.0) -> np.ndarray:
    mask = train_set.target_mask
    if not np.any(mask):
        return np.zeros(_regression_matrix(train_set).shape[1], dtype=np.float64)
    x = _regression_matrix(train_set)[mask]
    y = train_set.rul[mask].astype(np.float64)
    ident = np.eye(x.shape[1], dtype=np.float64)
    ident[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * ident, x.T @ y)


def fit_huber(train_set: BatteryWindowSet, delta: float = 8.0, iterations: int = 20) -> np.ndarray:
    mask = train_set.target_mask
    x = _regression_matrix(train_set)[mask]
    y = train_set.rul[mask].astype(np.float64)
    if not len(y):
        return np.zeros(_regression_matrix(train_set).shape[1], dtype=np.float64)
    beta = fit_ridge(train_set, alpha=10.0)
    for _ in range(iterations):
        residual = y - x @ beta
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1.0e-8))
        wx = x * weights[:, None]
        beta = np.linalg.solve(x.T @ wx + 1.0e-3 * np.eye(x.shape[1]), x.T @ (weights * y))
    return beta


def predict_regression(window_set: BatteryWindowSet, beta: np.ndarray) -> np.ndarray:
    return np.maximum(_regression_matrix(window_set) @ beta, 0.0).astype(np.float32)


def fit_soh_map(series: Sequence[BatterySeries]):
    xs, ys = [], []
    for item in series:
        if item.is_censored:
            continue
        xs.append(item.soh.astype(np.float64))
        ys.append(item.rul.astype(np.float64))
    if not xs:
        return lambda values: np.zeros(len(np.asarray(values)), dtype=np.float32)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = np.argsort(x)
    x, y = x[order], y[order]
    bins = np.linspace(float(x.min()), float(x.max()), 16)
    centers, values = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x <= hi)
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            values.append(float(np.median(y[mask])))
    if len(centers) < 2:
        centers, values = [float(x[0]), float(x[-1])], [float(y[0]), float(y[-1])]
    values = np.maximum.accumulate(np.asarray(values, dtype=np.float64))

    def mapper(soh: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(soh, dtype=np.float64), centers, values).astype(np.float32)

    return mapper


def predict_soh_map(series: BatterySeries, mapper) -> np.ndarray:
    return mapper(series.soh[-len(series) :])


def fit_capacity_slope_floor(series: Sequence[BatterySeries]) -> float:
    slopes = []
    for item in series:
        if len(item) >= 10:
            slopes.append(float(np.polyfit(np.arange(min(20, len(item))), item.capacity_clean[-min(20, len(item)) :], 1)[0]))
    negative = [-s for s in slopes if s < -1.0e-5]
    return float(np.median(negative)) if negative else 1.0e-3


def predict_capacity_trend(series: BatterySeries, slope_floor: float) -> np.ndarray:
    eol = float(series.eol_ah)
    output = np.zeros(len(series), dtype=np.float32)
    for i in range(len(series)):
        width = min(20, i + 1)
        if width >= 3:
            x = np.arange(width, dtype=np.float64)
            slope = float(np.polyfit(x, series.capacity_clean[i - width + 1 : i + 1], 1)[0])
        else:
            slope = 0.0
        magnitude = max(-slope, slope_floor)
        output[i] = max((series.capacity_clean[i] - eol) / magnitude, 0.0)
    return output


def _cell_window_sets(
    train_series: Sequence[BatterySeries],
    val_series: Sequence[BatterySeries],
    candidate: Candidate,
    *,
    common_endpoint: int = COMMON_ENDPOINT,
) -> tuple[BatteryWindowSet, BatteryWindowSet, object]:
    feature_names = select_features(candidate)
    scaler = fit_scaler(train_series, feature_names)
    train_set, _ = make_windows(
        train_series,
        candidate.seq_len,
        scaler=scaler,
        feature_names=feature_names,
        min_endpoint=common_endpoint,
    )
    val_set, _ = make_windows(
        val_series,
        candidate.seq_len,
        scaler=scaler,
        feature_names=feature_names,
        min_endpoint=common_endpoint,
    )
    return train_set, val_set, scaler


def _inner_cells(all_series: Sequence[BatterySeries], holdout: str) -> list[BatterySeries]:
    return [item for item in all_series if item.name != holdout]


def _choose_selection_rows(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not rows:
        raise ValueError("At least one candidate row is required")
    neural_row = min(rows, key=lambda row: float(row["mean_rmse"]))
    method_row = min(rows, key=lambda row: float(row["blend_mean_rmse"]))
    return neural_row, method_row


def _select_candidate_for_outer(
    series: Sequence[BatterySeries],
    outer_holdout: str,
    device: str,
    candidates: Sequence[Candidate],
    seeds: Sequence[int],
    *,
    epochs: int,
    common_endpoint: int,
) -> dict[str, object]:
    development = _inner_cells(series, outer_holdout)
    # A strict exact-RUL selector must use event-observed validation cells.  A
    # censored B0007 remains available for training survival constraints but is
    # never allowed to define an exact-RMSE architecture choice.
    validation_cells = [item for item in development if not item.is_censored]
    if not validation_cells:
        raise RuntimeError(f"No event-observed inner validation cells for {outer_holdout}")
    rows = []
    for candidate in candidates:
        fold_scores = []
        fold_epochs = []
        oof_model: dict[str, list[np.ndarray]] = {}
        oof_base: dict[str, list[np.ndarray]] = {}
        oof_validation: dict[str, list[BatteryWindowSet]] = {}
        for val in validation_cells:
            train_cells = [item for item in development if item.name != val.name]
            train_set, val_set, _ = _cell_window_sets(
                train_cells,
                [val],
                candidate,
                common_endpoint=common_endpoint,
            )
            model_preds = []
            for seed in seeds:
                model, best_epoch, _ = train_model(
                    candidate,
                    train_set,
                    val_set,
                    device,
                    seed,
                    epochs=epochs,
                )
                model_preds.append(predict_model(model, val_set, device))
                fold_epochs.append(best_epoch)
            model_pred = np.mean(np.stack(model_preds), axis=0)
            ridge_train, ridge_val, _ = _cell_window_sets(
                train_cells,
                [val],
                RIDGE_REFERENCE_CANDIDATE,
                common_endpoint=common_endpoint,
            )
            if not np.array_equal(ridge_val.endpoints, val_set.endpoints):
                raise RuntimeError("Ridge and neural validation endpoints are not aligned")
            ridge_pred = predict_regression(ridge_val, fit_ridge(ridge_train))
            fold_scores.append(rmse(val_set.rul[val_set.target_mask], model_pred[val_set.target_mask]))
            oof_model.setdefault(val.name, []).append(model_pred)
            oof_base.setdefault(val.name, []).append(ridge_pred)
            oof_validation.setdefault(val.name, []).append(val_set)
        score = float(np.mean(fold_scores))
        blend_alpha = _fit_blend_alpha(oof_model, oof_base, oof_validation)
        ridge_mean_rmse = float(np.mean([
            rmse(
                oof_validation[cell][0].rul[oof_validation[cell][0].target_mask],
                np.mean(np.stack(oof_base[cell]), axis=0)[oof_validation[cell][0].target_mask],
            )
            for cell in oof_validation
            if np.any(oof_validation[cell][0].target_mask)
        ]))
        blend_mean_rmse = float(np.mean([
            rmse(
                oof_validation[cell][0].rul[oof_validation[cell][0].target_mask],
                blend_alpha * np.mean(np.stack(oof_model[cell]), axis=0)[oof_validation[cell][0].target_mask]
                + (1.0 - blend_alpha) * np.mean(np.stack(oof_base[cell]), axis=0)[oof_validation[cell][0].target_mask],
            )
            for cell in oof_validation
            if np.any(oof_validation[cell][0].target_mask)
        ]))
        rows.append(
            {
                "candidate": candidate_dict(candidate),
                "inner_cell_rmse": {name: float(value) for name, value in zip([v.name for v in validation_cells], fold_scores)},
                "mean_rmse": score,
                "ridge_mean_rmse": ridge_mean_rmse,
                "blend_alpha": blend_alpha,
                "blend_mean_rmse": blend_mean_rmse,
                "blend_improvement_vs_ridge": ridge_mean_rmse - blend_mean_rmse,
                "median_best_epoch": int(np.median(fold_epochs)) if fold_epochs else epochs,
            }
        )
    # These are two separately predeclared algorithms.  The neural selector uses
    # neural-only RMSE; the complete-method selector uses the calibrated
    # neural/Ridge RMSE and may choose alpha=0 as a classical fallback.
    neural_row, method_row = _choose_selection_rows(rows)
    return {
        "outer_holdout": outer_holdout,
        "neural_selection": {
            "candidate": neural_row["candidate"],
            "mean_inner_rmse": neural_row["mean_rmse"],
            "fixed_epochs": neural_row["median_best_epoch"],
        },
        "method_selection": {
            "candidate": method_row["candidate"],
            "mean_inner_rmse": method_row["blend_mean_rmse"],
            "neural_inner_rmse": method_row["mean_rmse"],
            "ridge_inner_rmse": method_row["ridge_mean_rmse"],
            "improvement_vs_ridge": method_row["blend_improvement_vs_ridge"],
            "fixed_epochs": method_row["median_best_epoch"],
            "blend_alpha": float(method_row["blend_alpha"]),
        },
        "candidate_rows": rows,
        "selection_cells": [item.name for item in validation_cells],
        "common_endpoint": common_endpoint,
    }


def _train_final_family(
    candidate: Candidate,
    train_series: Sequence[BatterySeries],
    test_series: BatterySeries,
    device: str,
    seeds: Sequence[int],
    epochs: int,
    common_endpoint: int,
    scheduler_epoch_budget: int | None = None,
) -> tuple[dict[str, object], BatteryWindowSet, BatteryWindowSet]:
    feature_names = select_features(candidate)
    scaler = fit_scaler(train_series, feature_names)
    train_set, _ = make_windows(
        train_series,
        candidate.seq_len,
        scaler=scaler,
        feature_names=feature_names,
        min_endpoint=common_endpoint,
    )
    test_set, _ = make_windows(
        [test_series],
        candidate.seq_len,
        scaler=scaler,
        feature_names=feature_names,
        min_endpoint=common_endpoint,
    )
    seed_predictions = []
    parameter_count = None
    for seed in seeds:
        # Fixed-epoch final training does not inspect test labels.
        model, _, _ = train_model(
            candidate,
            train_set,
            None,
            device,
            seed,
            epochs=max(1, epochs),
            scheduler_epoch_budget=scheduler_epoch_budget,
        )
        if parameter_count is None:
            parameter_count = count_parameters(model)
        seed_predictions.append(predict_model(model, test_set, device))
    prediction_matrix = np.stack(seed_predictions, axis=0)
    seed_metrics = [
        _metrics_for_test(test_set, prediction, f"seed_{seed}")
        for seed, prediction in zip(seeds, prediction_matrix)
    ]
    return (
        {
            "candidate": candidate_dict(candidate),
            "parameter_count": int(parameter_count or 0),
            "seed_ids": [int(seed) for seed in seeds],
            "seed_predictions": prediction_matrix,
            "ensemble": np.mean(prediction_matrix, axis=0),
            "seed_metrics": seed_metrics,
            "seed_rmse": [metric["rmse"] for metric in seed_metrics],
            "scaler": scaler,
        },
        train_set,
        test_set,
    )


def _metrics_for_test(
    test_set: BatteryWindowSet,
    prediction: np.ndarray,
    name: str,
) -> dict[str, object]:
    exact = test_set.target_mask
    if np.any(exact):
        truth = test_set.rul[exact]
        pred = prediction[exact]
        return {
            "name": name,
            "status": "event_observed",
            "rmse": rmse(truth, pred),
            "mae": mae(truth, pred),
            "bias": bias(truth, pred),
            "nrmse_100_cycles": rmse(truth, pred) / NRMSE_CYCLES,
            "n_exact": int(exact.sum()),
            "n_total": int(len(test_set)),
            "stage_metrics": _stage_metrics(truth, pred),
        }
    violation = np.maximum(test_set.lower_bound_rul - prediction, 0.0)
    return {
        "name": name,
        "status": "right_censored",
        "rmse": None,
        "mae": None,
        "bias": None,
        "nrmse_100_cycles": None,
        "stage_metrics": {},
        "n_exact": 0,
        "n_total": int(len(test_set)),
        "mean_lower_bound_violation": float(np.mean(violation)) if len(violation) else None,
        "max_lower_bound_violation": float(np.max(violation)) if len(violation) else None,
    }


def _stage_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    if len(truth) == 0:
        return {}
    order = np.argsort(truth)
    bins = np.array_split(order, 3)
    labels = ("early", "mid", "late")
    return {
        label: {
            "rmse": rmse(truth[index], prediction[index]) if len(index) else None,
            "mae": mae(truth[index], prediction[index]) if len(index) else None,
            "bias": bias(truth[index], prediction[index]) if len(index) else None,
            "n": int(len(index)),
        }
        for label, index in zip(labels, bins)
    }


def _fit_blend_alpha(
    model_predictions: Mapping[str, list[np.ndarray]],
    baseline_predictions: Mapping[str, list[np.ndarray]],
    validation_sets: Mapping[str, list[BatteryWindowSet]],
) -> float:
    # Equal-cell objective; only inner OOF predictions reach this function.
    alpha_grid = np.arange(0.0, 1.0 + BLEND_ALPHA_STEP / 2.0, BLEND_ALPHA_STEP)
    scores = []
    for alpha in alpha_grid:
        cell_scores = []
        for cell, model_list in model_predictions.items():
            model_pred = np.mean(np.stack(model_list), axis=0)
            base_pred = np.mean(np.stack(baseline_predictions[cell]), axis=0)
            val = validation_sets[cell][0]
            mask = val.target_mask
            if np.any(mask):
                cell_scores.append(rmse(val.rul[mask], alpha * model_pred[mask] + (1 - alpha) * base_pred[mask]))
        scores.append(finite_mean(cell_scores))
    return float(alpha_grid[int(np.nanargmin(scores))])


def select_all(
    series: Sequence[BatterySeries],
    out_dir: Path,
    device: str,
    candidates: Sequence[Candidate],
    seeds: Sequence[int],
    final_seeds: Sequence[int],
    *,
    epochs: int,
    common_endpoint: int,
) -> dict[str, object]:
    if tuple(seeds) != tuple(final_seeds):
        raise ValueError(
            f"Selection and final seeds must match: selection={tuple(seeds)}, final={tuple(final_seeds)}"
        )
    selections = []
    for holdout in [item.name for item in series]:
        print(f"[select] outer={holdout}", flush=True)
        selections.append(
            _select_candidate_for_outer(
                series,
                holdout,
                device,
                candidates,
                seeds,
                epochs=epochs,
                common_endpoint=common_endpoint,
            )
        )
    config = {
        "protocol": "nested cell-LOOCV; strict14 exact labels; B0007 right-censored",
        "selection_seeds": list(seeds),
        "final_seeds": list(final_seeds),
        "candidates": [candidate_dict(c) for c in candidates],
        "fixed_families": list(FIXED_FAMILY_NAMES),
        "common_endpoint": common_endpoint,
        "selection_epoch_budget": int(epochs),
        "outer_selections": selections,
        "selection_objectives": {
            "selected_neural": "minimum inner-cell equal-fold neural ensemble RMSE",
            "selected_method_blend": "minimum inner-cell equal-fold neural/Ridge blend RMSE across candidate and alpha",
        },
        "blend_calibration": (
            f"inner-cell OOF alpha grid 0..1 step {BLEND_ALPHA_STEP:.2f} against Ridge; "
            "alpha=0 fallback allowed and frozen per outer fold"
        ),
        "censor_rule": "B0007 strict14 is right-censored; observed-record lower-bound hinge only",
        "nrmse_denominator_cycles": NRMSE_CYCLES,
        "nozzle_frozen": True,
    }
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    config["config_sha256"] = hashlib.sha256(payload).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_config.json", config)
    with (out_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "outer_holdout", "candidate", "model", "seq_len", "use_age", "use_capacity_aux",
            "mean_rmse", "ridge_mean_rmse", "blend_alpha", "blend_mean_rmse",
            "blend_improvement_vs_ridge", "median_best_epoch", "selected_neural",
            "selected_method_blend", "inner_cell_rmse",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for selection in selections:
            neural_chosen = selection["neural_selection"]["candidate"]["name"]
            method_chosen = selection["method_selection"]["candidate"]["name"]
            for row in selection["candidate_rows"]:
                candidate = row["candidate"]
                writer.writerow({
                    "outer_holdout": selection["outer_holdout"],
                    "candidate": candidate["name"],
                    "model": candidate["model"],
                    "seq_len": candidate["seq_len"],
                    "use_age": candidate["use_age"],
                    "use_capacity_aux": candidate["use_capacity_aux"],
                    "mean_rmse": row["mean_rmse"],
                    "ridge_mean_rmse": row["ridge_mean_rmse"],
                    "blend_alpha": row["blend_alpha"],
                    "blend_mean_rmse": row["blend_mean_rmse"],
                    "blend_improvement_vs_ridge": row["blend_improvement_vs_ridge"],
                    "median_best_epoch": row["median_best_epoch"],
                    "selected_neural": candidate["name"] == neural_chosen,
                    "selected_method_blend": candidate["name"] == method_chosen,
                    "inner_cell_rmse": json.dumps(row["inner_cell_rmse"], sort_keys=True),
                })
    return config


def _read_frozen_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = config.pop("config_sha256", None)
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        raise ValueError(f"Frozen battery config SHA256 mismatch: expected {expected}, computed {actual}")
    config["config_sha256"] = expected
    return config


def _candidate_from_config(value: Mapping[str, object]) -> Candidate:
    return Candidate(
        name=str(value["name"]),
        model=str(value["model"]),
        seq_len=int(value["seq_len"]),
        use_age=bool(value.get("use_age", True)),
        use_capacity_aux=bool(value.get("use_capacity_aux", True)),
    )


def _frozen_candidate_epoch(selection: Mapping[str, object], name: str) -> int:
    for row in selection["candidate_rows"]:
        if row["candidate"]["name"] == name:
            return int(row["median_best_epoch"])
    raise KeyError(f"Candidate {name!r} is absent from frozen selection")


def _macro_metrics(folds: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    exact = [
        next(metric for metric in fold["metrics"] if metric["name"] == method)
        for fold in folds
        if fold["status"] == "event_observed"
    ]
    return {
        "rmse": finite_mean(metric["rmse"] for metric in exact),
        "mae": finite_mean(metric["mae"] for metric in exact),
        "bias": finite_mean(metric["bias"] for metric in exact),
        "nrmse_100_cycles": finite_mean(metric.get("nrmse_100_cycles") for metric in exact),
        "stage_metrics": {
            stage: {
                key: finite_mean(
                    metric.get("stage_metrics", {}).get(stage, {}).get(key)
                    for metric in exact
                )
                for key in ("rmse", "mae", "bias")
            }
            for stage in ("early", "mid", "late")
        },
    }


def _final_all_legacy(
    raws: Mapping[str, RawBattery],
    series: Sequence[BatterySeries],
    config: Mapping[str, object],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
    *,
    common_endpoint: int,
) -> dict[str, object]:
    selections = {row["outer_holdout"]: row for row in config["outer_selections"]}
    frozen_seeds = tuple(int(seed) for seed in config.get("final_seeds", seeds))
    if tuple(seeds) != frozen_seeds:
        raise ValueError(f"Final seeds do not match frozen config: expected {frozen_seeds}, got {tuple(seeds)}")
    frozen_endpoint = int(config.get("common_endpoint", common_endpoint))
    if frozen_endpoint != common_endpoint:
        raise ValueError(f"Common endpoint does not match frozen config: expected {frozen_endpoint}, got {common_endpoint}")
    configured_candidates = {
        str(value["name"]): _candidate_from_config(value)
        for value in config.get("candidates", [])
    }
    missing_families = [name for name in FIXED_FAMILY_NAMES if name not in configured_candidates]
    if missing_families:
        raise ValueError(
            "Frozen config is missing fixed diagnostic families: "
            + ", ".join(missing_families)
        )
    fixed_candidates = {name: configured_candidates[name] for name in FIXED_FAMILY_NAMES}
    folds = []
    family_folds = []
    prediction_rows = []
    per_seed_rows = []
    per_seed_prediction_rows = []
    for holdout in [item.name for item in series]:
        outer_train = [item for item in series if item.name != holdout]
        selected = _candidate_from_config(selections[holdout]["selected"])
        epochs = int(selections[holdout]["fixed_epochs"])
        print(f"[final] outer={holdout} candidate={selected.name} epochs={epochs}", flush=True)
        model_info, train_set, test_set = _train_final_family(
            selected,
            outer_train,
            series[[item.name for item in series].index(holdout)],
            device,
            seeds,
            epochs,
            common_endpoint,
        )
        ridge_beta = fit_ridge(train_set)
        huber_beta = fit_huber(train_set)
        ridge_pred = predict_regression(test_set, ridge_beta)
        huber_pred = predict_regression(test_set, huber_beta)
        mapper = fit_soh_map(outer_train)
        test_series = series[[item.name for item in series].index(holdout)]
        soh_pred = mapper(test_series.soh[-len(test_set) :])
        slope_floor = fit_capacity_slope_floor(outer_train)
        trend_full = predict_capacity_trend(test_series, slope_floor)
        trend_pred = trend_full[test_set.endpoints]
        test_series = series[[item.name for item in series].index(holdout)]
        neural = model_info["ensemble"]
        metrics = [
            _metrics_for_test(test_set, neural, "selected_neural"),
            _metrics_for_test(test_set, ridge_pred, "ridge"),
            _metrics_for_test(test_set, huber_pred, "huber"),
            _metrics_for_test(test_set, soh_pred, "soh_monotone"),
            _metrics_for_test(test_set, trend_pred, "capacity_trend"),
        ]
        # The blend weight is frozen by inner-cell OOF calibration and is
        # therefore independent of all outer-test labels.
        blend_alpha = float(selections[holdout].get("blend_alpha", 1.0))
        blended = blend_alpha * neural + (1.0 - blend_alpha) * ridge_pred
        metrics.append(_metrics_for_test(test_set, blended, "primary_blend"))
        fold = {
            "holdout": holdout,
            "status": series[[item.name for item in series].index(holdout)].status,
            "candidate": candidate_dict(selected),
            "epochs": epochs,
            "parameter_count": model_info["parameter_count"],
            "n_train_windows": len(train_set),
            "n_test_windows": len(test_set),
            "seed_rmse": model_info["seed_rmse"],
            "seed_mean_rmse": finite_mean(model_info["seed_rmse"]),
            "blend_alpha_model": blend_alpha,
            "metrics": metrics,
            "scaler": model_info["scaler"].as_dict(),
        }
        for seed, metric, seed_prediction in zip(
            model_info["seed_ids"], model_info["seed_metrics"], model_info["seed_predictions"]
        ):
            per_seed_rows.append({
                "holdout": holdout,
                "family": "selected_neural",
                "candidate": selected.name,
                "seed": int(seed),
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "bias": metric["bias"],
                "nrmse_100_cycles": metric.get("nrmse_100_cycles"),
            })
            for index, prediction in enumerate(seed_prediction):
                per_seed_prediction_rows.append({
                    "holdout": holdout,
                    "family": "selected_neural",
                    "candidate": selected.name,
                    "seed": int(seed),
                    "endpoint": int(test_set.endpoints[index]),
                    "target_mask": bool(test_set.target_mask[index]),
                    "rul": None if not np.isfinite(test_set.rul[index]) else float(test_set.rul[index]),
                    "prediction": float(prediction),
                })
        # Fixed neural families are diagnostic baselines; they never alter the
        # frozen selected candidate or the primary blend.
        family_epoch = {
            name: _frozen_candidate_epoch(selections[holdout], name)
            for name in FIXED_FAMILY_NAMES
        }
        for family_name, family_candidate in fixed_candidates.items():
            family_info, _, family_test = _train_final_family(
                family_candidate,
                outer_train,
                test_series,
                device,
                seeds,
                family_epoch[family_name],
                common_endpoint,
            )
            family_metrics = [_metrics_for_test(family_test, family_info["ensemble"], family_name)]
            family_folds.append({
                "holdout": holdout,
                "status": series[[item.name for item in series].index(holdout)].status,
                "family": family_name,
                "candidate": candidate_dict(family_candidate),
                "epochs": family_epoch[family_name],
                "metrics": family_metrics,
                "seed_ids": family_info["seed_ids"],
                "seed_metrics": family_info["seed_metrics"],
            })
            for seed, metric, seed_prediction in zip(
                family_info["seed_ids"], family_info["seed_metrics"], family_info["seed_predictions"]
            ):
                per_seed_rows.append({
                    "holdout": holdout,
                    "family": family_name,
                    "candidate": family_name,
                    "seed": int(seed),
                    "rmse": metric["rmse"],
                    "mae": metric["mae"],
                    "bias": metric["bias"],
                    "nrmse_100_cycles": metric.get("nrmse_100_cycles"),
                })
                for index, prediction in enumerate(seed_prediction):
                    per_seed_prediction_rows.append({
                        "holdout": holdout,
                        "family": family_name,
                        "candidate": family_name,
                        "seed": int(seed),
                        "endpoint": int(family_test.endpoints[index]),
                        "target_mask": bool(family_test.target_mask[index]),
                        "rul": None if not np.isfinite(family_test.rul[index]) else float(family_test.rul[index]),
                        "prediction": float(prediction),
                    })
        folds.append(fold)
        for index in range(len(test_set)):
            row = {
                "holdout": holdout,
                "endpoint": int(test_set.endpoints[index]),
                "cell": str(test_set.cells[index]),
                "target_mask": bool(test_set.target_mask[index]),
                "rul": None if not np.isfinite(test_set.rul[index]) else float(test_set.rul[index]),
                "selected_neural": float(neural[index]),
                "ridge": float(ridge_pred[index]),
                "huber": float(huber_pred[index]),
                "soh_monotone": float(soh_pred[index]),
                "capacity_trend": float(trend_pred[index]),
                "primary_blend": float(blended[index]),
            }
            prediction_rows.append(row)
    exact_folds = [fold for fold in folds if fold["status"] == "event_observed"]
    summary_rows = []
    for fold in folds:
        for metric in fold["metrics"]:
            summary_rows.append({"holdout": fold["holdout"], **metric})
    methods = ["selected_neural", "ridge", "huber", "soh_monotone", "capacity_trend", "primary_blend"]
    macro_detail = {method: _macro_metrics(folds, method) for method in methods}
    family_macro = {
        family: _macro_metrics(
            [{"status": fold["status"], "metrics": fold["metrics"]} for fold in family_folds if fold["family"] == family],
            family,
        )
        for family in FIXED_FAMILY_NAMES
    }
    seed_groups: dict[tuple[str, int], list[float]] = {}
    for row in per_seed_rows:
        if row["holdout"] in {fold["holdout"] for fold in exact_folds} and row["rmse"] is not None:
            seed_groups.setdefault((str(row["family"]), int(row["seed"])), []).append(float(row["rmse"]))
    per_seed_summary = [
        {
            "family": family,
            "seed": seed,
            "mean_rmse": float(np.mean(values)),
            "std_rmse_across_folds": float(np.std(values)),
            "n_event_folds": len(values),
        }
        for (family, seed), values in sorted(seed_groups.items())
    ]
    result = {
        "protocol": "nested cell-LOOCV strict14; event-observed folds only in exact-RUL mean",
        "protocol_scope": "Three event-observed cells; B0007 strict14 is right-censored and excluded from exact-event macro",
        "folds": folds,
        "fixed_family_folds": family_folds,
        "exact_event_cells": [fold["holdout"] for fold in exact_folds],
        "mean_rmse_by_method": {method: detail["rmse"] for method, detail in macro_detail.items()},
        "macro_metrics_by_method": macro_detail,
        "fixed_family_macro_metrics": family_macro,
        "per_seed_summary": per_seed_summary,
        "final_seed_ids": list(seeds),
        "nozzle_frozen": True,
        "frozen_config_sha256": config["config_sha256"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "BATTERY_STRICT_REPORT.json", result)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]) if prediction_rows else ["holdout"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (out_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "holdout", "name", "status", "rmse", "mae", "bias", "nrmse_100_cycles",
            "n_exact", "n_total", "mean_lower_bound_violation", "max_lower_bound_violation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "candidate", "seed", "rmse", "mae", "bias", "nrmse_100_cycles"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (out_dir / "per_seed_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "candidate", "seed", "endpoint", "target_mask", "rul", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed_prediction_rows)
    return result


def final_all(
    raws: Mapping[str, RawBattery],
    series: Sequence[BatterySeries],
    config: Mapping[str, object],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
    *,
    common_endpoint: int,
) -> dict[str, object]:
    del raws
    selections = {row["outer_holdout"]: row for row in config["outer_selections"]}
    frozen_selection_seeds = tuple(int(seed) for seed in config.get("selection_seeds", seeds))
    frozen_final_seeds = tuple(int(seed) for seed in config.get("final_seeds", seeds))
    if tuple(seeds) != frozen_final_seeds or frozen_selection_seeds != frozen_final_seeds:
        raise ValueError(
            "Selection/final seeds do not match frozen v2 config: "
            f"selection={frozen_selection_seeds}, final={frozen_final_seeds}, requested={tuple(seeds)}"
        )
    frozen_endpoint = int(config.get("common_endpoint", common_endpoint))
    if frozen_endpoint != common_endpoint:
        raise ValueError(f"Common endpoint does not match frozen config: expected {frozen_endpoint}, got {common_endpoint}")
    scheduler_budget = int(config["selection_epoch_budget"])
    configured_candidates = {
        str(value["name"]): _candidate_from_config(value)
        for value in config.get("candidates", [])
    }
    missing_families = [name for name in FIXED_FAMILY_NAMES if name not in configured_candidates]
    if missing_families:
        raise ValueError("Frozen config is missing fixed diagnostic families: " + ", ".join(missing_families))
    fixed_candidates = {name: configured_candidates[name] for name in FIXED_FAMILY_NAMES}

    folds = []
    family_folds = []
    prediction_rows = []
    per_seed_rows = []
    per_seed_prediction_rows = []
    fixed_family_prediction_rows = []

    for test_series in series:
        holdout = test_series.name
        selection = selections[holdout]
        outer_train = [item for item in series if item.name != holdout]
        neural_spec = selection["neural_selection"]
        method_spec = selection["method_selection"]
        neural_candidate = _candidate_from_config(neural_spec["candidate"])
        method_candidate = _candidate_from_config(method_spec["candidate"])
        neural_epochs = int(neural_spec["fixed_epochs"])
        method_epochs = int(method_spec["fixed_epochs"])
        method_alpha = float(method_spec["blend_alpha"])
        if not 0.0 <= method_alpha <= 1.0:
            raise ValueError(f"Invalid frozen blend alpha for {holdout}: {method_alpha}")
        print(
            f"[final-v2] outer={holdout} neural={neural_candidate.name}/{neural_epochs} "
            f"method={method_candidate.name}/{method_epochs} alpha={method_alpha:.2f}",
            flush=True,
        )

        training_cache: dict[tuple[str, int], tuple[dict[str, object], BatteryWindowSet, BatteryWindowSet]] = {}

        def train_cached(candidate: Candidate, epochs: int):
            key = (candidate.name, int(epochs))
            if key not in training_cache:
                training_cache[key] = _train_final_family(
                    candidate,
                    outer_train,
                    test_series,
                    device,
                    seeds,
                    int(epochs),
                    common_endpoint,
                    scheduler_budget,
                )
            return training_cache[key]

        neural_info, neural_train, neural_test = train_cached(neural_candidate, neural_epochs)
        ridge_train, ridge_test, ridge_scaler = _cell_window_sets(
            outer_train,
            [test_series],
            RIDGE_REFERENCE_CANDIDATE,
            common_endpoint=common_endpoint,
        )
        if not np.array_equal(neural_test.endpoints, ridge_test.endpoints):
            raise RuntimeError(f"Neural/Ridge test endpoints are not aligned for {holdout}")
        ridge_pred = predict_regression(ridge_test, fit_ridge(ridge_train))
        huber_pred = predict_regression(ridge_test, fit_huber(ridge_train))
        mapper = fit_soh_map(outer_train)
        soh_pred = mapper(test_series.soh[ridge_test.endpoints])
        slope_floor = fit_capacity_slope_floor(outer_train)
        trend_pred = predict_capacity_trend(test_series, slope_floor)[ridge_test.endpoints]
        neural_pred = neural_info["ensemble"]

        method_info = None
        method_test = ridge_test
        if method_alpha > 0.0:
            method_info, _, method_test = train_cached(method_candidate, method_epochs)
            if not np.array_equal(method_test.endpoints, ridge_test.endpoints):
                raise RuntimeError(f"Method/Ridge test endpoints are not aligned for {holdout}")
            method_seed_predictions = method_info["seed_predictions"]
        else:
            method_seed_predictions = np.repeat(ridge_pred[None, :], len(seeds), axis=0)
        method_seed_blends = method_alpha * method_seed_predictions + (1.0 - method_alpha) * ridge_pred[None, :]
        method_pred = np.mean(method_seed_blends, axis=0)

        metrics = [
            _metrics_for_test(neural_test, neural_pred, "selected_neural"),
            _metrics_for_test(ridge_test, method_pred, "selected_method_blend"),
            _metrics_for_test(ridge_test, ridge_pred, "ridge"),
            _metrics_for_test(ridge_test, huber_pred, "huber"),
            _metrics_for_test(ridge_test, soh_pred, "soh_monotone"),
            _metrics_for_test(ridge_test, trend_pred, "capacity_trend"),
        ]
        fold = {
            "holdout": holdout,
            "status": test_series.status,
            "scheduler_epoch_budget": scheduler_budget,
            "neural_selection": {
                **neural_spec,
                "parameter_count": neural_info["parameter_count"],
                "seed_rmse": neural_info["seed_rmse"],
            },
            "method_selection": {
                **method_spec,
                "ridge_fallback": bool(method_alpha == 0.0),
                "neural_component_trained": bool(method_info is not None),
                "parameter_count": int(method_info["parameter_count"]) if method_info is not None else 0,
            },
            "n_train_windows": len(neural_train),
            "n_test_windows": len(ridge_test),
            "metrics": metrics,
            "neural_scaler": neural_info["scaler"].as_dict(),
            "ridge_scaler": ridge_scaler.as_dict(),
        }

        for seed, metric, seed_prediction in zip(
            neural_info["seed_ids"], neural_info["seed_metrics"], neural_info["seed_predictions"]
        ):
            per_seed_rows.append({
                "holdout": holdout,
                "family": "selected_neural",
                "candidate": neural_candidate.name,
                "seed": int(seed),
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "bias": metric["bias"],
                "nrmse_100_cycles": metric.get("nrmse_100_cycles"),
            })
            for index, prediction in enumerate(seed_prediction):
                per_seed_prediction_rows.append({
                    "holdout": holdout,
                    "family": "selected_neural",
                    "candidate": neural_candidate.name,
                    "seed": int(seed),
                    "endpoint": int(neural_test.endpoints[index]),
                    "target_mask": bool(neural_test.target_mask[index]),
                    "rul": None if not np.isfinite(neural_test.rul[index]) else float(neural_test.rul[index]),
                    "prediction": float(prediction),
                })
        for seed, seed_prediction in zip(seeds, method_seed_blends):
            metric = _metrics_for_test(ridge_test, seed_prediction, f"seed_{seed}")
            per_seed_rows.append({
                "holdout": holdout,
                "family": "selected_method_blend",
                "candidate": method_candidate.name,
                "seed": int(seed),
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "bias": metric["bias"],
                "nrmse_100_cycles": metric.get("nrmse_100_cycles"),
            })
            for index, prediction in enumerate(seed_prediction):
                per_seed_prediction_rows.append({
                    "holdout": holdout,
                    "family": "selected_method_blend",
                    "candidate": method_candidate.name,
                    "seed": int(seed),
                    "endpoint": int(ridge_test.endpoints[index]),
                    "target_mask": bool(ridge_test.target_mask[index]),
                    "rul": None if not np.isfinite(ridge_test.rul[index]) else float(ridge_test.rul[index]),
                    "prediction": float(prediction),
                })

        family_epoch = {
            name: _frozen_candidate_epoch(selection, name)
            for name in FIXED_FAMILY_NAMES
        }
        for family_name, family_candidate in fixed_candidates.items():
            family_info, _, family_test = train_cached(family_candidate, family_epoch[family_name])
            family_metric = _metrics_for_test(family_test, family_info["ensemble"], family_name)
            family_folds.append({
                "holdout": holdout,
                "status": test_series.status,
                "family": family_name,
                "candidate": candidate_dict(family_candidate),
                "epochs": family_epoch[family_name],
                "scheduler_epoch_budget": scheduler_budget,
                "metrics": [family_metric],
                "seed_ids": family_info["seed_ids"],
                "seed_metrics": family_info["seed_metrics"],
            })
            for index, prediction in enumerate(family_info["ensemble"]):
                fixed_family_prediction_rows.append({
                    "holdout": holdout,
                    "family": family_name,
                    "endpoint": int(family_test.endpoints[index]),
                    "target_mask": bool(family_test.target_mask[index]),
                    "rul": None if not np.isfinite(family_test.rul[index]) else float(family_test.rul[index]),
                    "prediction": float(prediction),
                })
            for seed, metric, seed_prediction in zip(
                family_info["seed_ids"], family_info["seed_metrics"], family_info["seed_predictions"]
            ):
                per_seed_rows.append({
                    "holdout": holdout,
                    "family": family_name,
                    "candidate": family_name,
                    "seed": int(seed),
                    "rmse": metric["rmse"],
                    "mae": metric["mae"],
                    "bias": metric["bias"],
                    "nrmse_100_cycles": metric.get("nrmse_100_cycles"),
                })
                for index, prediction in enumerate(seed_prediction):
                    per_seed_prediction_rows.append({
                        "holdout": holdout,
                        "family": family_name,
                        "candidate": family_name,
                        "seed": int(seed),
                        "endpoint": int(family_test.endpoints[index]),
                        "target_mask": bool(family_test.target_mask[index]),
                        "rul": None if not np.isfinite(family_test.rul[index]) else float(family_test.rul[index]),
                        "prediction": float(prediction),
                    })

        folds.append(fold)
        for index in range(len(ridge_test)):
            prediction_rows.append({
                "holdout": holdout,
                "endpoint": int(ridge_test.endpoints[index]),
                "cell": str(ridge_test.cells[index]),
                "target_mask": bool(ridge_test.target_mask[index]),
                "rul": None if not np.isfinite(ridge_test.rul[index]) else float(ridge_test.rul[index]),
                "selected_neural": float(neural_pred[index]),
                "selected_method_blend": float(method_pred[index]),
                "ridge": float(ridge_pred[index]),
                "huber": float(huber_pred[index]),
                "soh_monotone": float(soh_pred[index]),
                "capacity_trend": float(trend_pred[index]),
            })

    exact_folds = [fold for fold in folds if fold["status"] == "event_observed"]
    methods = ["selected_neural", "selected_method_blend", "ridge", "huber", "soh_monotone", "capacity_trend"]
    macro_detail = {method: _macro_metrics(folds, method) for method in methods}
    family_macro = {
        family: _macro_metrics(
            [{"status": fold["status"], "metrics": fold["metrics"]} for fold in family_folds if fold["family"] == family],
            family,
        )
        for family in FIXED_FAMILY_NAMES
    }
    event_names = {fold["holdout"] for fold in exact_folds}
    seed_groups: dict[tuple[str, int], list[float]] = {}
    for row in per_seed_rows:
        if row["holdout"] in event_names and row["rmse"] is not None:
            seed_groups.setdefault((str(row["family"]), int(row["seed"])), []).append(float(row["rmse"]))
    per_seed_summary = [
        {
            "family": family,
            "seed": seed,
            "mean_rmse": float(np.mean(values)),
            "std_rmse_across_folds": float(np.std(values)),
            "n_event_folds": len(values),
        }
        for (family, seed), values in sorted(seed_groups.items())
    ]
    result = {
        "schema": "battery_strict_v2_report",
        "protocol": "nested cell-LOOCV strict14; event-observed folds only in exact-RUL mean",
        "protocol_scope": "Three event-observed cells; B0007 strict14 is right-censored and excluded from exact-event macro",
        "selection_semantics": {
            "selected_neural": "candidate selected by inner neural RMSE",
            "selected_method_blend": "candidate and Ridge blend alpha selected by inner blend RMSE; alpha=0 fallback allowed",
        },
        "folds": folds,
        "fixed_family_folds": family_folds,
        "exact_event_cells": [fold["holdout"] for fold in exact_folds],
        "mean_rmse_by_method": {method: detail["rmse"] for method, detail in macro_detail.items()},
        "macro_metrics_by_method": macro_detail,
        "fixed_family_macro_metrics": family_macro,
        "per_seed_summary": per_seed_summary,
        "selection_and_final_seed_ids": list(seeds),
        "scheduler_epoch_budget": scheduler_budget,
        "nozzle_frozen": True,
        "frozen_config_sha256": config["config_sha256"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "BATTERY_STRICT_V2_REPORT.json", result)
    write_json(out_dir / "BATTERY_STRICT_REPORT.json", result)

    summary_rows = [
        {"holdout": fold["holdout"], "category": "main", **metric}
        for fold in folds for metric in fold["metrics"]
    ] + [
        {"holdout": fold["holdout"], "category": "fixed_family", **fold["metrics"][0]}
        for fold in family_folds
    ]
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]) if prediction_rows else ["holdout"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (out_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "holdout", "category", "name", "status", "rmse", "mae", "bias", "nrmse_100_cycles",
            "n_exact", "n_total", "mean_lower_bound_violation", "max_lower_bound_violation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "fixed_family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "status", "epochs", "scheduler_epoch_budget", "rmse", "mae", "bias", "nrmse_100_cycles"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for fold in family_folds:
            writer.writerow({**fold, **fold["metrics"][0]})
    with (out_dir / "fixed_family_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "endpoint", "target_mask", "rul", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fixed_family_prediction_rows)
    with (out_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "candidate", "seed", "rmse", "mae", "bias", "nrmse_100_cycles"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (out_dir / "per_seed_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "candidate", "seed", "endpoint", "target_mask", "rul", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed_prediction_rows)
    return result


def run_b0007_adaptive(
    raws: Mapping[str, RawBattery],
    strict_series: Sequence[BatterySeries],
    config: Mapping[str, object],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
) -> dict[str, object]:
    selections = {row["outer_holdout"]: row for row in config["outer_selections"]}
    selection = selections["B0007"]
    neural_spec = selection["neural_selection"]
    method_spec = selection["method_selection"]
    neural_candidate = _candidate_from_config(neural_spec["candidate"])
    method_candidate = _candidate_from_config(method_spec["candidate"])
    train_series = [item for item in strict_series if item.name != "B0007"]
    test_series = materialize_battery(raws["B0007"], "adaptive")
    endpoint = int(config["common_endpoint"])
    scheduler_budget = int(config["selection_epoch_budget"])
    neural_info, _, neural_test = _train_final_family(
        neural_candidate, train_series, test_series, device, seeds,
        int(neural_spec["fixed_epochs"]), endpoint, scheduler_budget,
    )
    ridge_train, ridge_test, _ = _cell_window_sets(
        train_series, [test_series], RIDGE_REFERENCE_CANDIDATE, common_endpoint=endpoint,
    )
    ridge_pred = predict_regression(ridge_test, fit_ridge(ridge_train))
    method_alpha = float(method_spec["blend_alpha"])
    if method_alpha == 0.0:
        method_pred = ridge_pred.copy()
    elif method_candidate == neural_candidate and int(method_spec["fixed_epochs"]) == int(neural_spec["fixed_epochs"]):
        method_pred = method_alpha * neural_info["ensemble"] + (1.0 - method_alpha) * ridge_pred
    else:
        method_info, _, method_test = _train_final_family(
            method_candidate, train_series, test_series, device, seeds,
            int(method_spec["fixed_epochs"]), endpoint, scheduler_budget,
        )
        if not np.array_equal(method_test.endpoints, ridge_test.endpoints):
            raise RuntimeError("Adaptive method/Ridge endpoints are not aligned")
        method_pred = method_alpha * method_info["ensemble"] + (1.0 - method_alpha) * ridge_pred
    result = {
        "schema": "battery_strict_v2_b0007_adaptive",
        "protocol": "B0007 adaptive test; strict14 selection frozen before adaptive evaluation",
        "holdout": "B0007",
        "status": test_series.status,
        "eol_ah": test_series.eol_ah,
        "life_cycle": test_series.life_cycle,
        "neural_selection": neural_spec,
        "method_selection": {**method_spec, "ridge_fallback": bool(method_alpha == 0.0)},
        "selected_neural_metrics": _metrics_for_test(neural_test, neural_info["ensemble"], "selected_neural"),
        "selected_method_blend_metrics": _metrics_for_test(ridge_test, method_pred, "selected_method_blend"),
        "ridge_metrics": _metrics_for_test(ridge_test, ridge_pred, "ridge"),
        "scheduler_epoch_budget": scheduler_budget,
        "nozzle_frozen": True,
        "frozen_config_sha256": config["config_sha256"],
    }
    write_json(out_dir / "B0007_ADAPTIVE_REPORT.json", result)
    return result


def _run_b0007_adaptive_legacy(
    raws: Mapping[str, RawBattery],
    strict_series: Sequence[BatterySeries],
    config: Mapping[str, object],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Evaluate fixed strict-trained configuration on B0007 rel80/adaptive EOL."""
    selections = {row["outer_holdout"]: row for row in config["outer_selections"]}
    selected_row = selections["B0007"]
    candidate = _candidate_from_config(selected_row["selected"])
    epochs = int(selected_row["fixed_epochs"])
    train_series = [item for item in strict_series if item.name != "B0007"]
    test_series = materialize_battery(raws["B0007"], "adaptive")
    frozen_endpoint = int(config.get("common_endpoint", COMMON_ENDPOINT))
    info, train_set, test_set = _train_final_family(
        candidate,
        train_series,
        test_series,
        device,
        seeds,
        epochs,
        frozen_endpoint,
    )
    neural = info["ensemble"]
    ridge_pred = predict_regression(test_set, fit_ridge(train_set))
    blend_alpha = float(selected_row.get("blend_alpha", 1.0))
    blended = blend_alpha * neural + (1.0 - blend_alpha) * ridge_pred
    metrics = _metrics_for_test(test_set, neural, "selected_neural")
    result = {
        "protocol": "B0007 adaptive test; B0005/B0006/B0018 strict14 training; fixed by strict inner selection",
        "holdout": "B0007",
        "candidate": candidate_dict(candidate),
        "epochs": epochs,
        "status": test_series.status,
        "eol_ah": test_series.eol_ah,
        "life_cycle": test_series.life_cycle,
        "metrics": metrics,
        "ridge_metrics": _metrics_for_test(test_set, ridge_pred, "ridge"),
        "blend_metrics": _metrics_for_test(test_set, blended, "primary_blend"),
        "blend_alpha_model": blend_alpha,
        "seed_rmse": info["seed_rmse"],
        "nozzle_frozen": True,
        "frozen_config_sha256": config["config_sha256"],
    }
    write_json(out_dir / "B0007_ADAPTIVE_REPORT.json", result)
    return result


def write_markdown(result: Mapping[str, object], adaptive: Mapping[str, object] | None, out_dir: Path) -> None:
    lines = [
        "# Strict Battery Benchmark",
        "",
        "- Protocol: nested cell-level LOOCV; strict14 exact labels.",
        "- Exact-event macro covers B0005/B0006/B0018 only; B0007 strict14 is right-censored.",
        "- `selected_neural` selects a neural candidate by inner neural RMSE.",
        "- `selected_method_blend` separately selects candidate and Ridge blend alpha by inner blend RMSE; alpha=0 is an explicit Ridge fallback.",
        f"- Selection and final use the same seeds; final retraining keeps the frozen scheduler budget of {result['scheduler_epoch_budget']} epochs.",
        "- Test labels are used only by the final evaluator.",
        "- Nozzle results are frozen and untouched.",
        "",
        "## Exact-Event Macro Mean",
        "",
        "| Method | RMSE | MAE | Bias | NRMSE / 100 cycles |",
        "|---|---:|---:|---:|---:|",
    ]
    details = result.get("macro_metrics_by_method", {})
    for name, value in result["mean_rmse_by_method"].items():
        detail = details.get(name, {})
        def fmt(item):
            return f"{item:.4f}" if item is not None and np.isfinite(item) else "N/A"
        lines.append(
            f"| {name} | {fmt(value)} | {fmt(detail.get('mae'))} | {fmt(detail.get('bias'))} | {fmt(detail.get('nrmse_100_cycles'))} |"
        )
    if result.get("fixed_family_macro_metrics"):
        lines.extend(["", "## Fixed Neural Families", "", "| Family | RMSE | MAE | Bias |", "|---|---:|---:|---:|"])
        for family, detail in result["fixed_family_macro_metrics"].items():
            lines.append(f"| {family} | {fmt(detail.get('rmse'))} | {fmt(detail.get('mae'))} | {fmt(detail.get('bias'))} |")
    lines.extend([
        "", "## Outer Folds", "",
        "| Holdout | Status | Neural candidate | Method candidate | Alpha | Ridge fallback | Neural RMSE | Method RMSE |",
        "|---|---|---|---|---:|---|---:|---:|",
    ])
    for fold in result["folds"]:
        neural = next(metric for metric in fold["metrics"] if metric["name"] == "selected_neural")
        method = next(metric for metric in fold["metrics"] if metric["name"] == "selected_method_blend")
        neural_name = fold["neural_selection"]["candidate"]["name"]
        method_name = fold["method_selection"]["candidate"]["name"]
        alpha = float(fold["method_selection"]["blend_alpha"])
        fallback = "yes" if fold["method_selection"]["ridge_fallback"] else "no"
        lines.append(
            f"| {fold['holdout']} | {fold['status']} | {neural_name} | {method_name} | "
            f"{alpha:.2f} | {fallback} | {fmt(neural['rmse'])} | {fmt(method['rmse'])} |"
        )
    if adaptive is not None:
        lines.extend([
            "",
            "## B0007 Adaptive Special",
            "",
            f"- EOL: {adaptive['eol_ah']:.6f} Ah",
            f"- Life cycle: {adaptive['life_cycle']:.1f}",
            f"- Neural candidate: `{adaptive['neural_selection']['candidate']['name']}`",
            f"- Method candidate: `{adaptive['method_selection']['candidate']['name']}`; alpha={adaptive['method_selection']['blend_alpha']:.2f}",
            f"- Neural RMSE: {adaptive['selected_neural_metrics']['rmse']:.4f} cycles",
            f"- Method RMSE: {adaptive['selected_method_blend_metrics']['rmse']:.4f} cycles",
            f"- Ridge RMSE: {adaptive['ridge_metrics']['rmse']:.4f} cycles",
            "",
            "This adaptive row has a separate EOL-derived capacity-margin feature distribution and is not mixed with strict14.",
        ])
    (out_dir / "BATTERY_STRICT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "final", "all"), default="all")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="outputs/battery_strict_v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(x) for x in FINAL_SEEDS))
    parser.add_argument("--selection-seeds", default=",".join(str(x) for x in SELECTION_SEEDS))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    final_seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    selection_seeds = tuple(int(x) for x in args.selection_seeds.split(",") if x.strip())
    candidates = CANDIDATES
    epochs = args.epochs
    if args.quick:
        # Keep every fixed diagnostic family available in quick replays so the
        # final audit path exercises the same declared family set.
        candidates = (CANDIDATES[0], CANDIDATES[1], CANDIDATES[2], CANDIDATES[4])
        selection_seeds = selection_seeds[:1]
        final_seeds = final_seeds[:1]
        epochs = min(epochs, 8)
    out_dir = Path(args.output)
    data_dir = Path(args.data_root) / "nasa_battery" / "5. Battery Data Set"
    print(f"Device={device} phase={args.phase} output={out_dir}", flush=True)
    raws = {name: load_battery_mat(data_dir / f"{name}.mat") for name in BATTERY_NAMES}
    strict_series = [materialize_battery(raws[name], "strict14") for name in BATTERY_NAMES]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.phase in ("select", "all"):
        config = select_all(
            strict_series,
            out_dir,
            device,
            candidates,
            selection_seeds,
            final_seeds,
            epochs=epochs,
            common_endpoint=COMMON_ENDPOINT,
        )
    else:
        config = _read_frozen_config(out_dir / "frozen_config.json")
    manifest = {
        "schema": "battery_strict_manifest_v3",
        "data_root": str(data_dir),
        "data_files": {name: sha256_file(data_dir / f"{name}.mat") for name in BATTERY_NAMES},
        "protocol": config.get("protocol", "strict14"),
        "features": list(DEFAULT_FEATURES),
        "common_endpoint": int(config.get("common_endpoint", COMMON_ENDPOINT)),
        "series": [describe_series(item) for item in strict_series],
        "selection_seeds": list(config.get("selection_seeds", selection_seeds)),
        "final_seeds": list(config.get("final_seeds", final_seeds)),
        "candidates": config.get("candidates", [candidate_dict(c) for c in candidates]),
        "fixed_families": config.get("fixed_families", list(FIXED_FAMILY_NAMES)),
        "selection_objectives": config.get("selection_objectives"),
        "selection_epoch_budget": config.get("selection_epoch_budget"),
        "blend_alpha_step": BLEND_ALPHA_STEP,
        "blend_calibration": config.get("blend_calibration"),
        "loss_and_censor_rule": config.get("censor_rule"),
        "metrics": {"nrmse_denominator_cycles": NRMSE_CYCLES, "stage_bins": "equal thirds by true RUL among event rows"},
        "baseline_definitions": ["Ridge", "Huber", "SOH monotone", "causal capacity trend"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config.get("config_sha256"),
        "quick": bool(args.quick),
        "quick_disclaimer": "Quick runs are smoke tests only and are not canonical results." if args.quick else None,
        "nozzle_frozen": True,
    }
    write_json(out_dir / "protocol_manifest.json", manifest)
    if args.phase in ("final", "all"):
        frozen_endpoint = int(config.get("common_endpoint", COMMON_ENDPOINT))
        result = final_all(
            raws,
            strict_series,
            config,
            out_dir,
            device,
            final_seeds,
            common_endpoint=frozen_endpoint,
        )
        adaptive = run_b0007_adaptive(raws, strict_series, config, out_dir, device, final_seeds)
        write_markdown(result, adaptive, out_dir)
        write_json(
            out_dir / "run_metadata.json",
            {
                "elapsed_sec": time.time() - t0,
                "phase": args.phase,
                "device": device,
                "schema": "battery_strict_v2_run_metadata",
                "selection_seeds": list(config.get("selection_seeds", selection_seeds)),
                "final_seeds": list(config.get("final_seeds", final_seeds)),
                "selection_epoch_budget": int(config.get("selection_epoch_budget", epochs)),
                "blend_alpha_step": BLEND_ALPHA_STEP,
                "config_sha256": config["config_sha256"],
                "nozzle_frozen": True,
            },
        )
        print(json.dumps(result["mean_rmse_by_method"], indent=2), flush=True)


if __name__ == "__main__":
    main()
