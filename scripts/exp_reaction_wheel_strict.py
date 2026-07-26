"""Strict FEMTO bearing benchmark for a reaction-wheel mechanical proxy.

This runner deliberately names the target a run-to-end ordinal RUL proxy.  The
FEMTO Learning_set used here has no official physical EOL/RUL label, so raw RUL
is retained-measurement index and is never described as spacecraft life.
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

from src.data.femto_strict import (
    BEARING_NAMES,
    FEATURE_GROUPS,
    FemtoSeries,
    FemtoWindowSet,
    describe_series,
    fit_scaler,
    load_femto_zip,
    make_windows,
)
from src.models.reaction_wheel_strict import build_reaction_wheel_model, count_parameters


SEEDS = (42, 123, 456, 2026, 3407)
COMMON_ENDPOINT = 19
SCHEDULER_EPOCH_BUDGET = 120
BASELINE_FEATURE_GROUP = "full"
BASELINE_SEQ_LEN = 20


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_group: str
    seq_len: int = 20


CANDIDATES = (
    Candidate("gru_base_l20", "gru", "base"),
    Candidate("ms_base_l20", "ms", "base"),
    Candidate("transformer_base_l20", "transformer", "base"),
    Candidate("ms_trend_l20", "ms", "base_trend"),
    Candidate("ms_full_l20", "ms", "full"),
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


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
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2))) if np.any(mask) else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yt[mask] - yp[mask]))) if np.any(mask) else float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(yp[mask] - yt[mask])) if np.any(mask) else float("nan")


def finite_mean(values: Iterable[float | None]) -> float:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            result.append(numeric)
    return float(np.mean(result)) if result else float("nan")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
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


def _feature_names(group: str) -> tuple[str, ...]:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown FEMTO feature group: {group}")
    return tuple(FEATURE_GROUPS[group])


def _series_scale(series: Sequence[FemtoSeries]) -> float:
    return max(max(float(item.raw_rul[0]) for item in series), 1.0)


# ---------------------------------------------------------------------------
# Windowing and model training
# ---------------------------------------------------------------------------


def _window_sets(
    train_series: Sequence[FemtoSeries],
    other_series: Sequence[FemtoSeries],
    candidate: Candidate,
    *,
    common_endpoint: int,
    rul_scale: float,
) -> tuple[FemtoWindowSet, FemtoWindowSet, object]:
    names = _feature_names(candidate.feature_group)
    scaler = fit_scaler(train_series, names)
    train_set = make_windows(
        train_series,
        candidate.seq_len,
        scaler,
        min_endpoint=common_endpoint,
        rul_scale=rul_scale,
    )
    other_set = make_windows(
        other_series,
        candidate.seq_len,
        scaler,
        min_endpoint=common_endpoint,
        rul_scale=rul_scale,
    )
    return train_set, other_set, scaler


def _append_cycle_pos(ws: FemtoWindowSet) -> FemtoWindowSet:
    """Append normalised cycle position as the last feature column.

    Ridge already has this information via `endpoints/rul_scale` in
    `_regression_matrix`.  Without it the GRU cannot learn absolute position
    in the degradation trajectory from the feature sequence alone when trained
    on only a handful of bearings.  Adding it puts both on equal footing.
    """
    # age: (N,) in [0, 1+ε]
    age = ws.endpoints.astype(np.float32) / max(float(ws.rul_scale), 1.0)
    # broadcast to (N, seq_len, 1) and concatenate along feature axis
    age_col = np.tile(age[:, None, None], (1, ws.X.shape[1], 1))
    return FemtoWindowSet(
        X=np.concatenate([ws.X, age_col], axis=-1).astype(np.float32),
        rul=ws.rul,
        life_fraction=ws.life_fraction,
        bearings=ws.bearings,
        endpoints=ws.endpoints,
        feature_names=ws.feature_names + ("cycle_pos_norm",),
        rul_scale=ws.rul_scale,
    )


def train_model(
    candidate: Candidate,
    train_set: FemtoWindowSet,
    val_set: FemtoWindowSet | None,
    device: str,
    seed: int,
    *,
    epochs: int,
    scheduler_epoch_budget: int,
    patience: int = 25,
    batch_size: int = 2048,
    learning_rate: float = 7.0e-4,
) -> tuple[object, int, float]:
    seed_everything(seed)
    model = build_reaction_wheel_model(candidate.model, train_set.X.shape[-1]).to(device)
    x = torch.as_tensor(train_set.X, dtype=torch.float32, device=device)
    target = torch.as_tensor(train_set.rul / train_set.rul_scale, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    batches = math.ceil(max(len(train_set), 1) / batch_size)
    schedule_steps = max(1, max(int(epochs), int(scheduler_epoch_budget)) * batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=schedule_steps)
    rng = np.random.default_rng(seed)
    val_x = val_y = None
    if val_set is not None:
        val_x = torch.as_tensor(val_set.X, dtype=torch.float32, device=device)
        val_y = val_set.rul
    best_state = None
    best_score = float("inf")
    best_epoch = int(epochs)
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_set))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            output = model(x[idx]).rul
            loss = F.smooth_l1_loss(output, target[idx], beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        if val_set is None:
            continue
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).rul.detach().cpu().numpy() * val_set.rul_scale
        score = rmse(val_y, val_pred)
        if score < best_score - 1.0e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_metadata = {
        "trained_epoch_limit": int(epochs),
        "scheduler_epoch_budget": int(scheduler_epoch_budget),
        "batches_per_epoch": int(batches),
    }
    model.eval()
    return model, best_epoch, float(best_score)


@torch.no_grad()
def predict_model(model, window_set: FemtoWindowSet, device: str) -> np.ndarray:
    x = torch.as_tensor(window_set.X, dtype=torch.float32, device=device)
    return (model(x).rul.detach().cpu().numpy() * window_set.rul_scale).astype(np.float32)


def _train_ensemble(
    candidate: Candidate,
    train_series: Sequence[FemtoSeries],
    test_series: Sequence[FemtoSeries],
    device: str,
    seeds: Sequence[int],
    epochs: int,
    *,
    common_endpoint: int,
    scheduler_epoch_budget: int,
) -> tuple[dict[str, object], FemtoWindowSet, FemtoWindowSet]:
    scale = _series_scale(train_series)
    train_set, test_set, scaler = _window_sets(
        train_series,
        test_series,
        candidate,
        common_endpoint=common_endpoint,
        rul_scale=scale,
    )
    train_set = _append_cycle_pos(train_set)
    test_set = _append_cycle_pos(test_set)
    seed_predictions = []
    seed_metrics = []
    parameter_count = None
    for seed in seeds:
        model, _, _ = train_model(
            candidate,
            train_set,
            None,
            device,
            int(seed),
            epochs=max(1, int(epochs)),
            scheduler_epoch_budget=scheduler_epoch_budget,
        )
        parameter_count = parameter_count or count_parameters(model)
        prediction = predict_model(model, test_set, device)
        seed_predictions.append(prediction)
        seed_metrics.append(_metrics_for(test_set, prediction, f"seed_{seed}", scale))
    matrix = np.stack(seed_predictions, axis=0)
    return (
        {
            "candidate": candidate_dict(candidate),
            "parameter_count": int(parameter_count or 0),
            "seed_ids": [int(seed) for seed in seeds],
            "seed_predictions": matrix,
            "ensemble": np.mean(matrix, axis=0),
            "seed_metrics": seed_metrics,
            "seed_rmse": [metric["rmse"] for metric in seed_metrics],
            "scaler": scaler,
            "rul_scale": scale,
        },
        train_set,
        test_set,
    )


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def _regression_matrix(window_set: FemtoWindowSet) -> np.ndarray:
    last = window_set.X[:, -1, :].astype(np.float64)
    age = window_set.endpoints.astype(np.float64)[:, None] / max(window_set.rul_scale, 1.0)
    return np.concatenate([np.ones((len(window_set), 1)), last, age], axis=1)


def fit_ridge(window_set: FemtoWindowSet, alpha: float = 10.0) -> np.ndarray:
    x = _regression_matrix(window_set)
    y = window_set.rul.astype(np.float64)
    ident = np.eye(x.shape[1], dtype=np.float64)
    ident[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * ident, x.T @ y)


def fit_huber(window_set: FemtoWindowSet, delta: float = 20.0, iterations: int = 20) -> np.ndarray:
    x = _regression_matrix(window_set)
    y = window_set.rul.astype(np.float64)
    beta = fit_ridge(window_set)
    for _ in range(iterations):
        residual = y - x @ beta
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1.0e-8))
        wx = x * weights[:, None]
        beta = np.linalg.solve(x.T @ wx + 1.0e-3 * np.eye(x.shape[1]), x.T @ (weights * y))
    return beta


def predict_regression(window_set: FemtoWindowSet, beta: np.ndarray) -> np.ndarray:
    return np.maximum(_regression_matrix(window_set) @ beta, 0.0).astype(np.float32)


def fit_mean(window_set: FemtoWindowSet) -> float:
    return float(np.mean(window_set.rul))


def predict_mean(window_set: FemtoWindowSet, value: float) -> np.ndarray:
    return np.full(len(window_set), value, dtype=np.float32)


def fit_time_trend(window_set: FemtoWindowSet) -> np.ndarray:
    x = np.column_stack([
        np.ones(len(window_set)),
        window_set.endpoints / max(window_set.rul_scale, 1.0),
        (window_set.endpoints / max(window_set.rul_scale, 1.0)) ** 2,
    ])
    return np.linalg.lstsq(x, window_set.rul, rcond=None)[0]


def predict_time_trend(window_set: FemtoWindowSet, beta: np.ndarray) -> np.ndarray:
    age = window_set.endpoints / max(window_set.rul_scale, 1.0)
    x = np.column_stack([np.ones(len(window_set)), age, age ** 2])
    return np.maximum(x @ beta, 0.0).astype(np.float32)


def _baseline_sets(train_series: Sequence[FemtoSeries], test_series: Sequence[FemtoSeries], common_endpoint: int):
    scale = _series_scale(train_series)
    baseline = Candidate("baseline_full_l20", "ridge", BASELINE_FEATURE_GROUP, BASELINE_SEQ_LEN)
    return _window_sets(train_series, test_series, baseline, common_endpoint=common_endpoint, rul_scale=scale)


# ---------------------------------------------------------------------------
# Metrics and nested selection
# ---------------------------------------------------------------------------


def _metrics_for(window_set: FemtoWindowSet, prediction: np.ndarray, name: str, train_scale: float) -> dict[str, object]:
    truth = window_set.rul
    raw = rmse(truth, prediction)
    test_scale = max(float(truth.max()), 1.0)
    return {
        "name": name,
        "rmse": raw,
        "mae": mae(truth, prediction),
        "bias": bias(truth, prediction),
        "train_normalized_rmse": raw / max(train_scale, 1.0),
        "life_fraction_rmse": rmse(truth / test_scale, prediction / test_scale),
        "legacy_test_normalized_rmse": raw / test_scale,
        "legacy_relative_rmse": raw / max(float(np.mean(truth)), 1.0e-8),
        "n": int(len(truth)),
        "train_rul_scale": float(train_scale),
        "test_rul_scale_descriptive": float(test_scale),
    }


def _select_outer(
    all_series: Sequence[FemtoSeries],
    holdout: str,
    candidates: Sequence[Candidate],
    device: str,
    seeds: Sequence[int],
    *,
    epochs: int,
    scheduler_epoch_budget: int,
    common_endpoint: int,
) -> dict[str, object]:
    development = [item for item in all_series if item.name != holdout]
    rows = []
    for candidate in candidates:
        scores = []
        epochs_seen = []
        per_validation = {}
        for val in development:
            fit = [item for item in development if item.name != val.name]
            scale = _series_scale(fit)
            train_set, val_set, _ = _window_sets(
                fit, [val], candidate, common_endpoint=common_endpoint, rul_scale=scale
            )
            train_set = _append_cycle_pos(train_set)
            val_set = _append_cycle_pos(val_set)
            predictions = []
            for seed in seeds:
                model, best_epoch, _ = train_model(
                    candidate,
                    train_set,
                    val_set,
                    device,
                    int(seed),
                    epochs=epochs,
                    scheduler_epoch_budget=scheduler_epoch_budget,
                )
                predictions.append(predict_model(model, val_set, device))
                epochs_seen.append(best_epoch)
            prediction = np.mean(np.stack(predictions), axis=0)
            score = rmse(val_set.rul, prediction)
            scores.append(score)
            per_validation[val.name] = score
        rows.append({
            "candidate": candidate_dict(candidate),
            "mean_inner_rmse": float(np.mean(scores)),
            "inner_bearing_rmse": {key: float(value) for key, value in per_validation.items()},
            "median_best_epoch": int(np.median(epochs_seen)) if epochs_seen else int(epochs),
            "n_inner_bearings": len(development),
        })
    selected = min(rows, key=lambda row: row["mean_inner_rmse"])
    return {
        "outer_holdout": holdout,
        "selected": selected["candidate"],
        "selected_mean_inner_rmse": selected["mean_inner_rmse"],
        "fixed_epochs": selected["median_best_epoch"],
        "candidate_rows": rows,
        "selection_bearings": [item.name for item in development],
        "common_endpoint": common_endpoint,
    }


def _macro_metrics(folds: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    metrics = [next(row for row in fold["metrics"] if row["name"] == method) for fold in folds]
    return {
        "rmse": finite_mean(row["rmse"] for row in metrics),
        "mae": finite_mean(row["mae"] for row in metrics),
        "bias": finite_mean(row["bias"] for row in metrics),
        "train_normalized_rmse": finite_mean(row["train_normalized_rmse"] for row in metrics),
        "life_fraction_rmse": finite_mean(row["life_fraction_rmse"] for row in metrics),
        "legacy_test_normalized_rmse": finite_mean(row["legacy_test_normalized_rmse"] for row in metrics),
        "legacy_relative_rmse": finite_mean(row["legacy_relative_rmse"] for row in metrics),
    }


def _bootstrap_macro(values: Sequence[float], seed: int = 2026, n_boot: int = 4000) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


# ---------------------------------------------------------------------------
# End-to-end benchmark
# ---------------------------------------------------------------------------


def select_all(
    series: Sequence[FemtoSeries],
    out_dir: Path,
    device: str,
    candidates: Sequence[Candidate],
    seeds: Sequence[int],
    *,
    epochs: int,
    scheduler_epoch_budget: int,
    common_endpoint: int,
) -> dict[str, object]:
    selections = []
    for holdout in [item.name for item in series]:
        print(f"[select] outer={holdout}", flush=True)
        selections.append(_select_outer(
            series, holdout, candidates, device, seeds,
            epochs=epochs,
            scheduler_epoch_budget=scheduler_epoch_budget,
            common_endpoint=common_endpoint,
        ))
    config = {
        "schema": "reaction_wheel_strict_config_v1",
        "protocol": "FEMTO Learning_set run-to-end ordinal RUL proxy; bearing-level nested LOOCV",
        "label_definition": "reverse retained-measurement index; final Learning_set record is endpoint",
        "endpoint_rule": "end_of_learning_run",
        "selection_seeds": list(seeds),
        "final_seeds": list(seeds),
        "candidates": [candidate_dict(candidate) for candidate in candidates],
        "baseline_feature_group": BASELINE_FEATURE_GROUP,
        "baseline_seq_len": BASELINE_SEQ_LEN,
        "common_endpoint": common_endpoint,
        "selection_epoch_budget": int(epochs),
        "scheduler_epoch_budget": int(scheduler_epoch_budget),
        "outer_selections": selections,
        "selection_objective": "equal-bearing inner validation raw ordinal-RUL RMSE",
        "normalization": "train-bearing RUL scale is headline; test-bearing scale is descriptive compatibility only",
        "test_label_usage": "test labels are read only by final evaluator; test-derived denominators are never selection inputs",
        "no_real_reaction_wheel_data": True,
        "nozzle_frozen": True,
    }
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    config["config_sha256"] = hashlib.sha256(payload).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_config.json", config)
    with (out_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "outer_holdout", "candidate", "model", "feature_group", "seq_len",
            "mean_inner_rmse", "median_best_epoch", "n_inner_bearings", "inner_bearing_rmse", "selected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for selection in selections:
            chosen = selection["selected"]["name"]
            for row in selection["candidate_rows"]:
                candidate = row["candidate"]
                writer.writerow({
                    "outer_holdout": selection["outer_holdout"],
                    "candidate": candidate["name"],
                    "model": candidate["model"],
                    "feature_group": candidate["feature_group"],
                    "seq_len": candidate["seq_len"],
                    "mean_inner_rmse": row["mean_inner_rmse"],
                    "median_best_epoch": row["median_best_epoch"],
                    "n_inner_bearings": row["n_inner_bearings"],
                    "inner_bearing_rmse": json.dumps(row["inner_bearing_rmse"], sort_keys=True),
                    "selected": candidate["name"] == chosen,
                })
    return config


def _read_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = config.pop("config_sha256", None)
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        raise ValueError(f"FEMTO frozen config SHA256 mismatch: expected {expected}, computed {actual}")
    config["config_sha256"] = expected
    return config


def _candidate(value: Mapping[str, object]) -> Candidate:
    return Candidate(
        name=str(value["name"]),
        model=str(value["model"]),
        feature_group=str(value["feature_group"]),
        seq_len=int(value.get("seq_len", BASELINE_SEQ_LEN)),
    )


def final_all(
    series: Sequence[FemtoSeries],
    config: Mapping[str, object],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
) -> dict[str, object]:
    frozen_seeds = tuple(int(seed) for seed in config["final_seeds"])
    if tuple(seeds) != frozen_seeds or tuple(config["selection_seeds"]) != frozen_seeds:
        raise ValueError("Selection and final seed sets must match frozen FEMTO config")
    if int(config["common_endpoint"]) != COMMON_ENDPOINT:
        raise ValueError("FEMTO common endpoint does not match runner constant")
    selections = {row["outer_holdout"]: row for row in config["outer_selections"]}
    configured = {_candidate(value).name: _candidate(value) for value in config["candidates"]}
    folds = []
    prediction_rows = []
    per_seed_rows = []
    per_seed_prediction_rows = []
    fixed_folds = []
    fixed_prediction_rows = []
    methods = ["selected_neural", "mean_baseline", "time_trend", "ridge", "huber"]
    for test_series in series:
        holdout = test_series.name
        outer_train = [item for item in series if item.name != holdout]
        selection = selections[holdout]
        candidate = _candidate(selection["selected"])
        epochs = int(selection["fixed_epochs"])
        print(f"[final] outer={holdout} candidate={candidate.name} epochs={epochs}", flush=True)
        neural_info, train_set, test_set = _train_ensemble(
            candidate, outer_train, [test_series], device, seeds, epochs,
            common_endpoint=COMMON_ENDPOINT,
            scheduler_epoch_budget=int(config["scheduler_epoch_budget"]),
        )
        baseline_train, baseline_test, baseline_scaler = _baseline_sets(outer_train, [test_series], COMMON_ENDPOINT)
        mean_pred = predict_mean(baseline_test, fit_mean(baseline_train))
        trend_pred = predict_time_trend(baseline_test, fit_time_trend(baseline_train))
        ridge_pred = predict_regression(baseline_test, fit_ridge(baseline_train))
        huber_pred = predict_regression(baseline_test, fit_huber(baseline_train))
        metrics = [
            _metrics_for(test_set, neural_info["ensemble"], "selected_neural", neural_info["rul_scale"]),
            _metrics_for(baseline_test, mean_pred, "mean_baseline", baseline_test.rul_scale),
            _metrics_for(baseline_test, trend_pred, "time_trend", baseline_test.rul_scale),
            _metrics_for(baseline_test, ridge_pred, "ridge", baseline_test.rul_scale),
            _metrics_for(baseline_test, huber_pred, "huber", baseline_test.rul_scale),
        ]
        fold = {
            "holdout": holdout,
            "status": "event_observed_end_of_run",
            "candidate": candidate_dict(candidate),
            "epochs": epochs,
            "scheduler_epoch_budget": int(config["scheduler_epoch_budget"]),
            "parameter_count": neural_info["parameter_count"],
            "n_train_windows": len(train_set),
            "n_test_windows": len(test_set),
            "metrics": metrics,
            "seed_rmse": neural_info["seed_rmse"],
            "scaler": neural_info["scaler"].as_dict(),
            "baseline_scaler": baseline_scaler.as_dict(),
        }
        for seed, metric, prediction in zip(neural_info["seed_ids"], neural_info["seed_metrics"], neural_info["seed_predictions"]):
            per_seed_rows.append({
                "holdout": holdout,
                "family": "selected_neural",
                "candidate": candidate.name,
                "seed": int(seed),
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "bias": metric["bias"],
            })
            for index, value in enumerate(prediction):
                per_seed_prediction_rows.append({
                    "holdout": holdout,
                    "family": "selected_neural",
                    "candidate": candidate.name,
                    "seed": int(seed),
                    "endpoint": int(test_set.endpoints[index]),
                    "rul": float(test_set.rul[index]),
                    "prediction": float(value),
                })
        for index in range(len(test_set)):
            prediction_rows.append({
                "holdout": holdout,
                "endpoint": int(test_set.endpoints[index]),
                "rul": float(test_set.rul[index]),
                "selected_neural": float(neural_info["ensemble"][index]),
                "mean_baseline": float(mean_pred[index]),
                "time_trend": float(trend_pred[index]),
                "ridge": float(ridge_pred[index]),
                "huber": float(huber_pred[index]),
            })
        folds.append(fold)

        for fixed_name, fixed_candidate in configured.items():
            fixed_epochs = next(
                int(row["median_best_epoch"])
                for row in selection["candidate_rows"]
                if row["candidate"]["name"] == fixed_name
            )
            fixed_info, _, fixed_test = _train_ensemble(
                fixed_candidate, outer_train, [test_series], device, seeds, fixed_epochs,
                common_endpoint=COMMON_ENDPOINT,
                scheduler_epoch_budget=int(config["scheduler_epoch_budget"]),
            )
            fixed_metric = _metrics_for(fixed_test, fixed_info["ensemble"], fixed_name, fixed_info["rul_scale"])
            fixed_folds.append({
                "holdout": holdout,
                "family": fixed_name,
                "candidate": candidate_dict(fixed_candidate),
                "epochs": fixed_epochs,
                "scheduler_epoch_budget": int(config["scheduler_epoch_budget"]),
                "metric": fixed_metric,
                "seed_metrics": fixed_info["seed_metrics"],
            })
            for index, value in enumerate(fixed_info["ensemble"]):
                fixed_prediction_rows.append({
                    "holdout": holdout,
                    "family": fixed_name,
                    "endpoint": int(fixed_test.endpoints[index]),
                    "rul": float(fixed_test.rul[index]),
                    "prediction": float(value),
                })
    macro = {method: _macro_metrics(folds, method) for method in methods}
    fixed_macro = {
        family: _macro_metrics(
            [{"metrics": [row["metric"]]} for row in fixed_folds if row["family"] == family],
            family,
        )
        for family in configured
    }
    pooled = {}
    for method in methods:
        truth = np.asarray([row["rul"] for row in prediction_rows], dtype=np.float64)
        pred = np.asarray([row[method] for row in prediction_rows], dtype=np.float64)
        pooled[method] = {"rmse": rmse(truth, pred), "mae": mae(truth, pred), "bias": bias(truth, pred)}
    bootstrap = {
        method: _bootstrap_macro(
            [next(item for item in fold["metrics"] if item["name"] == method)["rmse"] for fold in folds]
        )
        for method in methods
    }
    result = {
        "schema": "reaction_wheel_strict_v1_report",
        "domain": "femto_bearing_reaction_wheel_proxy",
        "proxy_scope": "FEMTO/PRONOSTIA Learning_set run-to-end ordinal RUL; not physical reaction-wheel life",
        "label_definition": config["label_definition"],
        "endpoint_rule": config["endpoint_rule"],
        "protocol": config["protocol"],
        "folds": folds,
        "fixed_family_folds": fixed_folds,
        "exact_event_cells": [fold["holdout"] for fold in folds],
        "mean_rmse_by_method": {method: details["rmse"] for method, details in macro.items()},
        "macro_metrics_by_method": macro,
        "pooled_metrics_by_method": pooled,
        "fixed_family_macro_metrics": fixed_macro,
        "bearing_bootstrap_95": bootstrap,
        "seed_ids": list(seeds),
        "scheduler_epoch_budget": int(config["scheduler_epoch_budget"]),
        "frozen_config_sha256": config["config_sha256"],
        "no_real_reaction_wheel_data": True,
        "nozzle_frozen": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "REACTION_WHEEL_STRICT_REPORT.json", result)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (out_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "name", "rmse", "mae", "bias", "train_normalized_rmse", "life_fraction_rmse", "legacy_test_normalized_rmse", "legacy_relative_rmse", "n"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for fold in folds:
            writer.writerows({"holdout": fold["holdout"], **metric} for metric in fold["metrics"])
    with (out_dir / "fixed_family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "epochs", "scheduler_epoch_budget", "rmse", "mae", "bias", "train_normalized_rmse", "life_fraction_rmse"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in fixed_folds:
            writer.writerow({"holdout": row["holdout"], "family": row["family"], "epochs": row["epochs"], "scheduler_epoch_budget": row["scheduler_epoch_budget"], **row["metric"]})
    with (out_dir / "fixed_family_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "endpoint", "rul", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fixed_prediction_rows)
    with (out_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "candidate", "seed", "rmse", "mae", "bias"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (out_dir / "per_seed_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout", "family", "candidate", "seed", "endpoint", "rul", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed_prediction_rows)
    return result


def write_markdown(result: Mapping[str, object], out_dir: Path) -> None:
    def fmt(value):
        return f"{value:.4f}" if value is not None and np.isfinite(value) else "N/A"
    lines = [
        "# Strict FEMTO Reaction-Wheel Proxy Benchmark",
        "",
        "- This is a FEMTO/PRONOSTIA bearing run-to-end ordinal RUL proxy, not real reaction-wheel telemetry.",
        "- Labels are retained-measurement indices; the final Learning_set record is the endpoint by protocol.",
        "- Outer split is leave-one-bearing-out; inner selection rotates validation across all development bearings.",
        "- Scaler, checkpoint, feature group and epoch selection use training/development bearings only.",
        "- Headline normalized RMSE uses the train-bearing RUL scale. Test-derived normalization is descriptive only.",
        "",
        "## Bearing-Macro Metrics",
        "",
        "| Method | RMSE | MAE | Bias | Train-normalized RMSE | Life-fraction RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, details in result["macro_metrics_by_method"].items():
        lines.append(f"| {name} | {fmt(details['rmse'])} | {fmt(details['mae'])} | {fmt(details['bias'])} | {fmt(details['train_normalized_rmse'])} | {fmt(details['life_fraction_rmse'])} |")
    lines.extend(["", "## Pooled Metrics", "", "| Method | RMSE | MAE | Bias |", "|---|---:|---:|---:|"])
    for name, details in result["pooled_metrics_by_method"].items():
        lines.append(f"| {name} | {fmt(details['rmse'])} | {fmt(details['mae'])} | {fmt(details['bias'])} |")
    lines.extend(["", "## Outer Bearings", "", "| Holdout | Candidate | RMSE | Train-normalized RMSE |", "|---|---|---:|---:|"])
    for fold in result["folds"]:
        metric = next(row for row in fold["metrics"] if row["name"] == "selected_neural")
        lines.append(f"| {fold['holdout']} | {fold['candidate']['name']} | {fmt(metric['rmse'])} | {fmt(metric['train_normalized_rmse'])} |")
    lines.extend(["", "## Bootstrap", "", "| Method | Mean | 95% lower | 95% upper |", "|---|---:|---:|---:|"])
    for name, details in result["bearing_bootstrap_95"].items():
        lines.append(f"| {name} | {fmt(details['mean'])} | {fmt(details['lower_95'])} | {fmt(details['upper_95'])} |")
    (out_dir / "REACTION_WHEEL_STRICT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--data-zip", default=None)
    parser.add_argument("--output", default="outputs/reaction_wheel_strict_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--epochs", type=int, default=SCHEDULER_EPOCH_BUDGET)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    epochs = int(args.epochs)
    candidates = CANDIDATES
    if args.quick:
        seeds = seeds[:1]
        epochs = min(epochs, 8)
    data_zip = Path(args.data_zip) if args.data_zip else Path(args.data_root) / "femto_bearing.zip"
    out_dir = Path(args.output)
    print(f"Device={device} output={out_dir} data={data_zip}", flush=True)
    series = load_femto_zip(data_zip, feature_group="full", max_files_per_bearing=None)
    ordered = [series[name] for name in BEARING_NAMES if name in series]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    config_path = out_dir / "frozen_config.json"
    if config_path.exists() and not args.quick:
        config = _read_config(config_path)
    else:
        config = select_all(
            ordered,
            out_dir,
            device,
            candidates,
            seeds,
            epochs=epochs,
            scheduler_epoch_budget=epochs,
            common_endpoint=COMMON_ENDPOINT,
        )
    manifest = {
        "schema": "reaction_wheel_strict_manifest_v1",
        "data_zip": str(data_zip),
        "data_zip_sha256": sha256_file(data_zip),
        "protocol": config["protocol"],
        "label_definition": config["label_definition"],
        "endpoint_rule": config["endpoint_rule"],
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "series": [describe_series(item) for item in ordered],
        "selection_seeds": config["selection_seeds"],
        "final_seeds": config["final_seeds"],
        "candidates": config["candidates"],
        "common_endpoint": config["common_endpoint"],
        "scheduler_epoch_budget": config["scheduler_epoch_budget"],
        "headline_normalization": config["normalization"],
        "test_label_usage": config["test_label_usage"],
        "quick": bool(args.quick),
        "quick_disclaimer": "Quick runs are smoke tests only and are not canonical results." if args.quick else None,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config["config_sha256"],
        "no_real_reaction_wheel_data": True,
        "nozzle_frozen": True,
    }
    write_json(out_dir / "protocol_manifest.json", manifest)
    if not args.quick or config_path.exists():
        result = final_all(ordered, config, out_dir, device, seeds)
        write_markdown(result, out_dir)
        write_json(out_dir / "run_metadata.json", {
            "schema": "reaction_wheel_strict_run_metadata_v1",
            "elapsed_sec": time.time() - t0,
            "device": device,
            "seed_ids": list(seeds),
            "config_sha256": config["config_sha256"],
            "nozzle_frozen": True,
        })
        print(json.dumps(result["mean_rmse_by_method"], indent=2), flush=True)


if __name__ == "__main__":
    main()
