"""Auditable FEMTO/IMS -> COMSOL target outer-fold transfer.

Each COMSOL trajectory is used exactly once as an outer holdout.  For every
holdout and shot count, one deterministic remaining trajectory is validation
only and the first N trajectories in the remaining ordered pool are used for
target fitting.  The holdout is never used for scaling, model selection, or
early stopping.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_femto_ims_to_comsol import (  # noqa: E402
    METHODS,
    SEEDS,
    SOURCE_EPOCHS,
    TARGET_EPOCHS,
    TARGET_FEATURE_TIER,
    TARGET_SEQ_LEN,
    PreparedData,
    SourceBundle,
    _json_safe,
    _natural_group,
    _new_target_model,
    _predict_target,
    _select_ridge_alpha,
    _target_stats,
    _train_loop,
    build_source_bundle,
    copy_shared_encoder,
    count_parameters,
    metrics,
    rmse,
    seed_everything,
    sha256_file,
    train_source_model,
    write_json,
)
from src.data.reaction_wheel_sim import SimulationArchive, SimulationTable, load_simulation_archive  # noqa: E402
from src.models.comsol_sequence_baselines import build_comsol_sequence_baseline  # noqa: E402


TARGET_N_SHOTS = (1, 2, 3)
TREND_METHODS = ("trend_linear", "trend_quadratic", "trend_huber")
SEQUENCE_METHODS = ("target_gru", "target_tcn", "target_transformer")
EXTRA_BASELINE_METHODS = TREND_METHODS + SEQUENCE_METHODS
ENSEMBLE_METHODS = METHODS + EXTRA_BASELINE_METHODS


def macro_rows_per_source_fold_shot(include_extra_baselines: bool, seeds: int) -> int:
    """Return the exact macro-row count for one source, outer fold, and N."""
    base = 3 * int(seeds) + 1 + 4
    if not include_extra_baselines:
        return base
    trend = 2 * len(TREND_METHODS)
    sequence = len(SEQUENCE_METHODS) * (int(seeds) + 1)
    return base + trend + sequence


def outer_schedule(groups: Sequence[str], n_shots: int) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return a deterministic, disjoint outer-fold schedule."""
    ordered = tuple(sorted((str(group) for group in groups), key=_natural_group))
    if len(ordered) < 4:
        raise ValueError("Outer-fold COMSOL protocol requires at least four trajectories")
    schedule: dict[str, dict[str, tuple[str, ...]]] = {}
    for index, holdout in enumerate(ordered):
        remaining = tuple(group for group in ordered if group != holdout)
        validation = (remaining[index % len(remaining)],)
        fit_pool = tuple(group for group in remaining if group not in validation)
        if len(fit_pool) < n_shots:
            raise ValueError(f"Not enough fit groups for N={n_shots}, holdout={holdout}")
        schedule[holdout] = {
            "fit_groups": fit_pool[:n_shots],
            "validation_groups": validation,
            "test_groups": (holdout,),
        }
    return schedule


def _make_target_windows(
    table: SimulationTable,
    groups: Sequence[str],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    label_scale: float,
) -> PreparedData:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    names: list[str] = []
    endpoints: list[int] = []
    contexts: list[np.ndarray] = []
    for group in groups:
        mask = table.group_ids == group
        rows = np.asarray(
            [int(float(row["周期编号"])) for row in np.asarray(table.rows, dtype=object)[mask]],
            dtype=np.int64,
        )
        order = np.argsort(rows)
        features = table.features[mask][order]
        target = table.target[mask][order]
        values = (features.astype(np.float32) - mean) / std
        # The first observed operating point is available at deployment and
        # carries trajectory-specific initial-condition information. Repeat it
        # as context for every causal window; no target labels or EOL values
        # participate in this transformation.
        initial_context = values[0].copy()
        for endpoint in range(TARGET_SEQ_LEN - 1, len(values)):
            xs.append(values[endpoint - TARGET_SEQ_LEN + 1 : endpoint + 1])
            ys.append(float(target[endpoint]) / label_scale)
            names.append(group)
            endpoints.append(endpoint)
            contexts.append(initial_context)
    if not xs:
        raise ValueError(f"Target groups {groups} are too short for sequence length {TARGET_SEQ_LEN}")
    return PreparedData(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.float32),
        units=np.asarray(names),
        endpoints=np.asarray(endpoints, dtype=np.int64),
        scale=float(label_scale),
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        context=np.stack(contexts).astype(np.float32),
    )


def prepare_outer_target(
    table: SimulationTable,
    *,
    fit_groups: Sequence[str],
    validation_groups: Sequence[str],
    test_groups: Sequence[str],
) -> tuple[PreparedData, PreparedData, PreparedData, dict[str, object]]:
    fit_set = set(fit_groups)
    validation_set = set(validation_groups)
    test_set = set(test_groups)
    if fit_set & validation_set or fit_set & test_set or validation_set & test_set:
        raise ValueError("Outer-fold fit, validation, and test groups must be disjoint")
    mean, std, label_scale = _target_stats(table, tuple(fit_groups))
    train = _make_target_windows(table, fit_groups, mean=mean, std=std, label_scale=label_scale)
    validation = _make_target_windows(table, validation_groups, mean=mean, std=std, label_scale=label_scale)
    test = _make_target_windows(table, test_groups, mean=mean, std=std, label_scale=label_scale)
    audit = {
        "fit_groups": list(fit_groups),
        "validation_groups": list(validation_groups),
        "test_groups": list(test_groups),
        "feature_names": list(table.feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "label_scale_from_fit_groups": label_scale,
        "holdout_labels_used_for_fit": False,
        "holdout_labels_used_for_validation": False,
        "holdout_in_scaler_fit": bool(test_set & fit_set),
        "holdout_in_validation": bool(test_set & validation_set),
    }
    return train, validation, test, audit


def _trend_design(data: PreparedData, degree: int, endpoint_scale: float) -> np.ndarray:
    time = data.endpoints.astype(np.float64) / max(float(endpoint_scale), 1.0)
    return np.column_stack([np.ones(len(time), dtype=np.float64), *[time ** power for power in range(1, degree + 1)]])


def _fit_trend(data: PreparedData, degree: int, endpoint_scale: float, alpha: float = 0.0) -> np.ndarray:
    design = _trend_design(data, degree, endpoint_scale)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + float(alpha) * penalty,
        design.T @ data.y.astype(np.float64),
    )


def _predict_trend(beta: np.ndarray, data: PreparedData, degree: int, endpoint_scale: float) -> np.ndarray:
    normalized = np.maximum(_trend_design(data, degree, endpoint_scale) @ beta, 0.0)
    return (normalized.astype(np.float32) * data.scale).astype(np.float32)


def _select_quadratic_trend(train: PreparedData, validation: PreparedData) -> tuple[np.ndarray, dict[str, object]]:
    endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
    best: tuple[float, np.ndarray, float] | None = None
    for alpha in (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0):
        beta = _fit_trend(train, degree=2, endpoint_scale=endpoint_scale, alpha=alpha)
        prediction = _predict_trend(beta, validation, degree=2, endpoint_scale=endpoint_scale)
        score = rmse(validation.y * validation.scale, prediction)
        if best is None or score < best[2]:
            best = (alpha, beta, score)
    if best is None:
        raise RuntimeError("Quadratic trend selection produced no candidate")
    alpha, beta, score = best
    return beta, {
        "endpoint_scale_from_fit_groups": endpoint_scale,
        "selected_alpha": float(alpha),
        "validation_raw_rmse": float(score),
    }


def _fit_huber_trend(
    data: PreparedData,
    endpoint_scale: float,
    delta: float,
    *,
    max_iterations: int = 50,
) -> np.ndarray:
    design = _trend_design(data, degree=1, endpoint_scale=endpoint_scale)
    target = data.y.astype(np.float64)
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(max_iterations):
        residual = target - design @ beta
        median = float(np.median(residual))
        scale = max(float(1.4826 * np.median(np.abs(residual - median))), 1.0e-4)
        cutoff = max(float(delta) * scale, 1.0e-4)
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1.0e-12))
        weighted_design = design * weights[:, None]
        updated = np.linalg.solve(
            weighted_design.T @ design + 1.0e-8 * np.eye(design.shape[1], dtype=np.float64),
            weighted_design.T @ target,
        )
        if float(np.max(np.abs(updated - beta))) < 1.0e-8:
            beta = updated
            break
        beta = updated
    return beta


def _select_huber_trend(train: PreparedData, validation: PreparedData) -> tuple[np.ndarray, dict[str, object]]:
    endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
    best: tuple[float, np.ndarray, float] | None = None
    for delta in (0.75, 1.0, 1.35, 2.0):
        beta = _fit_huber_trend(train, endpoint_scale, delta)
        prediction = _predict_trend(beta, validation, degree=1, endpoint_scale=endpoint_scale)
        score = rmse(validation.y * validation.scale, prediction)
        if best is None or score < best[2]:
            best = (delta, beta, score)
    if best is None:
        raise RuntimeError("Huber trend selection produced no candidate")
    delta, beta, score = best
    return beta, {
        "endpoint_scale_from_fit_groups": endpoint_scale,
        "selected_delta": float(delta),
        "validation_raw_rmse": float(score),
        "irls_iterations_max": 50,
    }


def _new_sequence_baseline(name: str, n_features: int, seed: int) -> torch.nn.Module:
    seed_everything(seed)
    key = name.removeprefix("target_")
    return build_comsol_sequence_baseline(key, n_features, sequence_length=TARGET_SEQ_LEN)


def _train_sequence_baseline(
    model: torch.nn.Module,
    train: PreparedData,
    validation: PreparedData,
    *,
    device: str,
    seed: int,
    epochs: int,
    patience: int = 15,
) -> tuple[torch.nn.Module, dict[str, object]]:
    seed_everything(seed)
    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=7.0e-4, weight_decay=1.0e-4)
    batch_size = min(2048, max(16, len(train.x)))
    rng = np.random.default_rng(seed)
    train_x = torch.as_tensor(train.x, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train.y, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(validation.x, dtype=torch.float32, device=device)
    validation_y = validation.y.astype(np.float32)
    best_score = float("inf")
    best_epoch = int(epochs)
    best_state = None
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train.x))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            output = model(train_x[index]).rul
            loss = F.smooth_l1_loss(output, train_y[index], beta=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation_x).rul.detach().cpu().numpy()
        score = rmse(validation_y, prediction)
        if score < best_score - 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_validation_normalized_rmse": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "model_initialization_seed": int(seed),
    }


@torch.no_grad()
def _predict_sequence_baseline(model: torch.nn.Module, data: PreparedData, device: str) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    prediction = model(tensor).rul.detach().cpu().numpy()
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Sequence baseline produced non-finite predictions")
    return (np.maximum(prediction, 0.0).astype(np.float32) * data.scale).astype(np.float32)


def summarize_outer_folds(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize ensemble predictions across held-out COMSOL trajectories."""
    summaries: list[dict[str, object]] = []
    source_names = sorted({str(row["source"]) for row in rows})
    for source_name in source_names:
        for n_shots in TARGET_N_SHOTS:
            selected = [
                row for row in rows
                if row["source"] == source_name and row["n_shots"] == n_shots and str(row["method"]).endswith("_ensemble")
            ]
            scratch = {
                str(row["outer_holdout"]): row
                for row in selected
                if row["method"] == "scratch_ensemble"
            }
            for method in (f"{name}_ensemble" for name in ENSEMBLE_METHODS):
                method_rows = [row for row in selected if row["method"] == method]
                if not method_rows:
                    continue
                fold_metrics = np.asarray([float(row["raw_rmse"]) for row in method_rows], dtype=np.float64)
                baseline = np.asarray(
                    [float(scratch[str(row["outer_holdout"])]["raw_rmse"]) for row in method_rows],
                    dtype=np.float64,
                )
                macro_rmse = float(np.mean(fold_metrics))
                scratch_macro_rmse = float(np.mean(baseline))
                improvement = float(100.0 * (scratch_macro_rmse - macro_rmse) / max(scratch_macro_rmse, 1.0e-12))
                wins = int(np.sum(fold_metrics < baseline))
                folds = len(method_rows)
                positive_evidence = (
                    method == "transfer_ensemble"
                    and improvement >= 5.0
                    and wins > folds / 2.0
                )
                summaries.append({
                    "source": source_name,
                    "n_shots": n_shots,
                    "method": method,
                    "outer_folds": folds,
                    "macro_raw_rmse": macro_rmse,
                    "macro_mae": float(np.mean([float(row["mae"]) for row in method_rows])),
                    "macro_bias": float(np.mean([float(row["bias"]) for row in method_rows])),
                    "macro_normalized_rmse": float(np.mean([float(row["normalized_rmse"]) for row in method_rows])),
                    "scratch_macro_raw_rmse": scratch_macro_rmse,
                    "improvement_vs_scratch_pct": improvement,
                    "outer_fold_wins_vs_scratch": wins,
                    "outer_fold_losses_or_ties_vs_scratch": int(folds - wins),
                    "positive_transfer_evidence": positive_evidence,
                    "acceptance_reason": (
                        "macro improvement >=5% and majority of outer folds beat scratch"
                        if positive_evidence
                        else (
                            "diagnostic: does not meet both >=5% macro improvement and majority-fold win criteria"
                            if method == "transfer_ensemble"
                            else "benchmark comparator; positive-transfer acceptance criterion applies only to transfer_ensemble"
                        )
                    ),
                })
    return summaries


def run_target_outerloo(
    source_name: str,
    source_bundle: SourceBundle,
    source_models: dict[int, torch.nn.Module],
    table: SimulationTable,
    *,
    seeds: Sequence[int],
    epochs: int,
    device: str,
    holdouts: Sequence[str] | None = None,
    include_extra_baselines: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    groups = tuple(sorted(set(table.group_ids.tolist()), key=_natural_group))
    selected = tuple(holdouts) if holdouts is not None else groups
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for holdout in selected:
        for n_shots in TARGET_N_SHOTS:
            split = outer_schedule(groups, n_shots)[holdout]
            train, validation, test, audit = prepare_outer_target(
                table,
                fit_groups=split["fit_groups"],
                validation_groups=split["validation_groups"],
                test_groups=split["test_groups"],
            )
            audit["outer_holdout"] = holdout
            audit["n_shots"] = n_shots
            audits[f"{holdout}:{n_shots}"] = audit
            target_methods = list(METHODS)
            if include_extra_baselines:
                target_methods.extend(EXTRA_BASELINE_METHODS)
            method_predictions: dict[str, list[np.ndarray]] = {
                method: [] for method in target_methods if method != "ridge"
            }
            for method in ("transfer", "frozen_encoder", "scratch"):
                for seed in seeds:
                    training_seed = int(seed) + n_shots * 1000 + int(holdout) * 10000
                    model = _new_target_model(
                        source_models.get(seed),
                        len(source_bundle.feature_names),
                        len(table.feature_names),
                        method,
                        device,
                        initialization_seed=training_seed,
                    )
                    model, fit_diag = _train_loop(
                        model,
                        train.x,
                        train.y,
                        validation.x,
                        validation.y,
                        device=device,
                        seed=training_seed,
                        epochs=epochs,
                        source=False,
                        frozen_encoder=method == "frozen_encoder",
                    )
                    fit_diag["model_initialization_seed"] = training_seed
                    prediction = _predict_target(model, test, device=device)
                    method_predictions[method].append(prediction)
                    rows.append({
                        "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                        "method": method, **metrics(test.y * test.scale, prediction, test.scale, method, seed),
                        "fit_groups": list(split["fit_groups"]),
                        "validation_groups": list(split["validation_groups"]),
                        "test_groups": list(split["test_groups"]),
                        "fit_diagnostics": fit_diag,
                    })
            ridge_alpha, ridge_validation_rmse = _select_ridge_alpha(train, validation)
            from scripts.exp_femto_ims_to_comsol import _fit_ridge, _predict_ridge
            beta = _fit_ridge(train, ridge_alpha)
            ridge_prediction = _predict_ridge(test, beta)
            method_predictions["ridge"] = [ridge_prediction]
            rows.append({
                "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                "method": "ridge",
                **metrics(test.y * test.scale, ridge_prediction, test.scale, "ridge", None),
                "fit_groups": list(split["fit_groups"]),
                "validation_groups": list(split["validation_groups"]),
                "test_groups": list(split["test_groups"]),
                "ridge_alpha": ridge_alpha, "ridge_validation_rmse": ridge_validation_rmse,
            })
            if include_extra_baselines:
                endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
                linear_beta = _fit_trend(train, degree=1, endpoint_scale=endpoint_scale)
                linear_prediction = _predict_trend(linear_beta, test, degree=1, endpoint_scale=endpoint_scale)
                method_predictions["trend_linear"] = [linear_prediction]
                rows.append({
                    "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                    "method": "trend_linear",
                    **metrics(test.y * test.scale, linear_prediction, test.scale, "trend_linear", None),
                    "fit_groups": list(split["fit_groups"]),
                    "validation_groups": list(split["validation_groups"]),
                    "test_groups": list(split["test_groups"]),
                    "fit_diagnostics": {
                        "family": "ordinary_least_squares_time_trend",
                        "degree": 1,
                        "endpoint_scale_from_fit_groups": endpoint_scale,
                    },
                })
                quadratic_beta, quadratic_diag = _select_quadratic_trend(train, validation)
                quadratic_prediction = _predict_trend(
                    quadratic_beta,
                    test,
                    degree=2,
                    endpoint_scale=float(quadratic_diag["endpoint_scale_from_fit_groups"]),
                )
                method_predictions["trend_quadratic"] = [quadratic_prediction]
                rows.append({
                    "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                    "method": "trend_quadratic",
                    **metrics(test.y * test.scale, quadratic_prediction, test.scale, "trend_quadratic", None),
                    "fit_groups": list(split["fit_groups"]),
                    "validation_groups": list(split["validation_groups"]),
                    "test_groups": list(split["test_groups"]),
                    "fit_diagnostics": {"family": "ridge_quadratic_time_trend", **quadratic_diag},
                })
                huber_beta, huber_diag = _select_huber_trend(train, validation)
                huber_prediction = _predict_trend(
                    huber_beta,
                    test,
                    degree=1,
                    endpoint_scale=float(huber_diag["endpoint_scale_from_fit_groups"]),
                )
                method_predictions["trend_huber"] = [huber_prediction]
                rows.append({
                    "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                    "method": "trend_huber",
                    **metrics(test.y * test.scale, huber_prediction, test.scale, "trend_huber", None),
                    "fit_groups": list(split["fit_groups"]),
                    "validation_groups": list(split["validation_groups"]),
                    "test_groups": list(split["test_groups"]),
                    "fit_diagnostics": {"family": "huber_linear_time_trend", **huber_diag},
                })
                for method_index, method in enumerate(SEQUENCE_METHODS, start=1):
                    for seed in seeds:
                        training_seed = (
                            int(seed)
                            + n_shots * 1000
                            + int(holdout) * 10000
                            + method_index * 100000
                        )
                        model = _new_sequence_baseline(method, len(table.feature_names), training_seed)
                        model, fit_diag = _train_sequence_baseline(
                            model,
                            train,
                            validation,
                            device=device,
                            seed=training_seed,
                            epochs=epochs,
                        )
                        prediction = _predict_sequence_baseline(model, test, device)
                        method_predictions[method].append(prediction)
                        rows.append({
                            "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                            "method": method,
                            **metrics(test.y * test.scale, prediction, test.scale, method, seed),
                            "fit_groups": list(split["fit_groups"]),
                            "validation_groups": list(split["validation_groups"]),
                            "test_groups": list(split["test_groups"]),
                            "fit_diagnostics": {
                                "family": "target_only_sequence_model",
                                "architecture": method.removeprefix("target_"),
                                **fit_diag,
                            },
                        })
            truth = test.y * test.scale
            for method, method_values in method_predictions.items():
                aggregate = np.mean(np.stack(method_values), axis=0)
                rows.append({
                    "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                    "method": f"{method}_ensemble",
                    **metrics(truth, aggregate, test.scale, f"{method}_ensemble", None),
                    "fit_groups": list(split["fit_groups"]),
                    "validation_groups": list(split["validation_groups"]),
                    "test_groups": list(split["test_groups"]),
                })
                for endpoint, actual, predicted in zip(test.endpoints, truth, aggregate):
                    predictions.append({
                        "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                        "method": f"{method}_ensemble", "target_group": holdout,
                        "endpoint": int(endpoint), "true_rul": float(actual), "prediction": float(predicted),
                    })
    return rows, predictions, audits


def run(source_names: Sequence[str], args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir) if args.temp_dir else output.parent / ".outerloo_temp"
    if not temp_dir.is_absolute():
        temp_dir = (PROJECT_ROOT / temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(temp_dir)
    started = time.time()
    try:
        device = args.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(min(max(1, int(args.cpu_threads)), torch.get_num_threads()))
        seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
        if args.quick:
            seeds = seeds[:2]
        source_epochs = min(args.source_epochs, 8) if args.quick else args.source_epochs
        target_epochs = min(args.target_epochs, 8) if args.quick else args.target_epochs
        max_files = args.source_max_files if not args.quick else (args.source_max_files or 120)
        comsol_path = Path(args.comsol_archive)
        source_paths = {"femto": Path(args.femto_zip), "ims": Path(args.ims_path)}
        missing = [str(path) for path in [comsol_path, *(source_paths[name] for name in source_names)] if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
        archive: SimulationArchive = load_simulation_archive(comsol_path)
        table = archive.aligned[TARGET_FEATURE_TIER]
        groups = tuple(sorted(set(table.group_ids.tolist()), key=_natural_group))
        if len(groups) != 5:
            raise ValueError(f"COMSOL outer protocol requires five groups, found {groups}")
        if any(set(outer_schedule(groups, n)) != set(groups) for n in TARGET_N_SHOTS):
            raise ValueError("Outer schedule does not cover every trajectory")
        holdouts = tuple(args.holdouts.split(",")) if args.holdouts.strip() else groups
        if any(group not in groups for group in holdouts):
            raise ValueError(f"Unknown holdout in {holdouts}; available groups are {groups}")
        all_rows: list[dict[str, object]] = []
        all_predictions: list[dict[str, object]] = []
        source_reports: dict[str, object] = {}
        for source_name in source_names:
            bundle = build_source_bundle(source_name, {}, source_paths[source_name], max_files)
            source_models: dict[int, torch.nn.Module] = {}
            source_training = []
            for seed in seeds:
                model, diagnostic = train_source_model(
                    bundle, target_features=len(table.feature_names), seed=seed,
                    epochs=source_epochs, device=device,
                )
                source_models[seed] = copy.deepcopy(model).cpu()
                source_training.append({"seed": seed, **diagnostic})
            rows, prediction_rows, audits = run_target_outerloo(
                source_name, bundle, source_models, table, seeds=seeds,
                epochs=target_epochs, device=device, holdouts=holdouts,
                include_extra_baselines=bool(args.include_extra_baselines),
            )
            all_rows.extend(rows)
            all_predictions.extend(prediction_rows)
            source_reports[source_name] = {
                "manifest": {
                    "source": source_name, "sha256": bundle.sha256,
                    "unit_names": sorted(bundle.units), "source_train_units": list(bundle.train_names),
                    "source_validation_units": list(bundle.validation_names),
                    "feature_names": list(bundle.feature_names), "source_label_scale": bundle.train.scale,
                },
                "source_training": source_training,
                "target_audit_by_outer_fold": audits,
                "source_model_parameter_count": count_parameters(source_models[seeds[0]]),
            }
        preflight = {
            "schema": f"femto_ims_to_comsol_outerloo_preflight_v{3 if args.include_extra_baselines else 2}",
            "comsol_archive": str(comsol_path), "comsol_sha256": archive.sha256,
            "comsol_feature_tier": TARGET_FEATURE_TIER, "comsol_groups": list(groups),
            "outer_holdouts": list(holdouts), "target_scaler_fit_only_on_fit_groups": True,
            "holdout_labels_used_for_selection": False, "holdout_labels_used_for_fit": False,
            "source_names": list(source_names),
            "include_extra_baselines": bool(args.include_extra_baselines),
            "extra_baseline_methods": list(EXTRA_BASELINE_METHODS) if args.include_extra_baselines else [],
            "execution_device": device,
            "cpu_threads": torch.get_num_threads() if device == "cpu" else None,
            "finite_checks": bool(np.all(np.isfinite(table.features)) and np.all(np.isfinite(table.target))),
        }
        protocol = {
            "schema": f"femto_ims_to_comsol_outerloo_v{3 if args.include_extra_baselines else 2}",
            "source_arms": list(source_names), "target_feature_tier": TARGET_FEATURE_TIER,
            "target_protocol": "five COMSOL trajectories outer-LOO; deterministic validation rotation; N fit groups",
            "outer_holdouts": list(holdouts), "n_shots": list(TARGET_N_SHOTS),
            "schedule": {str(n): outer_schedule(groups, n) for n in TARGET_N_SHOTS},
            "target_scaler": "fit on fit_groups only", "target_label_scale": "fit on fit_groups only",
            "transfer": "pretrained shared encoder plus target projection/head, all trainable",
            "frozen_ablation": "pretrained shared encoder frozen; target projection/head trainable",
            "scratch": "same target architecture, data, epoch budget, early stopping and seeds",
            "ridge": "target-only baseline with validation-selected alpha",
            "additional_baselines": (
                {
                    "trend_linear": "ordinary least-squares target RUL versus causal endpoint; fit groups only",
                    "trend_quadratic": "quadratic target RUL trend with validation-selected ridge penalty",
                    "trend_huber": "robust Huber linear target RUL trend with validation-selected delta",
                    "target_gru": "target-only GRU with the same target windows, epochs, validation and seeds",
                    "target_tcn": "target-only causal multi-scale TCN with the same target windows, epochs, validation and seeds",
                    "target_transformer": "target-only Transformer encoder with the same target windows, epochs, validation and seeds",
                }
                if args.include_extra_baselines
                else {}
            ),
            "seeds": list(seeds), "source_epochs": source_epochs, "target_epochs": target_epochs,
            "execution_device": device, "cpu_threads": torch.get_num_threads() if device == "cpu" else None,
            "final_holdout_evaluation_once": True,
            "proxy_warning": "COMSOL is a reaction-wheel simulation domain, not real flight telemetry",
        }
        write_json(output / "PREFLIGHT.json", preflight)
        write_json(output / "PROTOCOL.json", protocol)
        write_json(output / "DATA_MANIFEST.json", {
            "schema": f"femto_ims_to_comsol_outerloo_data_manifest_v{3 if args.include_extra_baselines else 2}",
            "inputs": {"comsol": {"path": str(comsol_path), "sha256": archive.sha256, "groups": list(groups)},
                       "sources": {name: source_reports[name]["manifest"] for name in source_names}},
            "outer_split": protocol["schedule"], "holdout_labels_used_for_selection": False,
        })
        outer_fold_summary = summarize_outer_folds(all_rows)
        report = {"schema": f"femto_ims_to_comsol_outerloo_report_v{3 if args.include_extra_baselines else 2}", "protocol": protocol,
                  "preflight": preflight, "sources": source_reports, "macro_rows": all_rows,
                  "outer_fold_summary": outer_fold_summary, "elapsed_sec": time.time() - started}
        write_json(output / "REPORT.json", report)
        with (output / "MACRO.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in all_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(_json_safe(all_rows))
        with (output / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["source", "n_shots", "outer_holdout", "method", "target_group", "endpoint", "true_rul", "prediction"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_predictions)
        with (output / "OUTER_FOLD_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in outer_fold_summary for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(_json_safe(outer_fold_summary))
        write_json(output / "ACCEPTANCE.json", {
            "schema": f"femto_ims_to_comsol_outerloo_acceptance_v{3 if args.include_extra_baselines else 2}",
            "criterion": "transfer macro RMSE improves >=5% over scratch and wins a majority of outer folds",
            "results": outer_fold_summary,
        })
        return {"output": str(output), "sources": list(source_names), "rows": len(all_rows),
                "outer_folds": len(holdouts), "elapsed_sec": time.time() - started}
    finally:
        tempfile.tempdir = previous_tempdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("femto", "ims", "both"), default="both")
    parser.add_argument("--femto-zip", default="data/processed/femto_bearing.zip")
    parser.add_argument("--ims-path", default="data/processed/ims_processed")
    parser.add_argument("--comsol-archive", default="data/raw/competition/reaction_wheel_comsol_degradation.zip")
    parser.add_argument("--output", default="outputs/femto_ims_to_comsol_outerloo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--source-epochs", type=int, default=SOURCE_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=TARGET_EPOCHS)
    parser.add_argument("--source-max-files", type=int, default=None)
    parser.add_argument("--holdouts", default="")
    parser.add_argument("--temp-dir", default="", help="Temporary extraction directory; defaults beside output.")
    parser.add_argument(
        "--include-extra-baselines",
        action="store_true",
        help="Run three target trend and three target sequence comparator families under the outer-fold protocol.",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    source_names = ("femto", "ims") if args.source == "both" else (args.source,)
    print(json.dumps(run(source_names, args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
