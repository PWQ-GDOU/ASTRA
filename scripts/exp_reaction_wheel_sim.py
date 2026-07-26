"""Initial RUL model for the supplied COMSOL reaction-wheel simulation data.

This is a fixed trajectory split experiment: tracks 1-3 train, track 4
validates candidate/model epochs, and track 5 is evaluated once at the end.
Only operational aligned-cycle features are used; oracle state is excluded.
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

from src.data.reaction_wheel_sim import SimulationTable, load_simulation_archive
from src.models.reaction_wheel_strict import build_reaction_wheel_model, count_parameters


SEEDS = (42, 123, 456, 2026, 3407)
SEQ_LEN = 20
EPOCH_BUDGET = 120


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str


CANDIDATES = (
    Candidate("gru_operational_l20", "gru"),
    Candidate("ms_operational_l20", "ms"),
    Candidate("transformer_operational_l20", "transformer"),
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
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if np.any(mask) else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if np.any(mask) else float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(y_pred[mask] - y_true[mask])) if np.any(mask) else float("nan")


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
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_rows(table: SimulationTable, group: str) -> tuple[np.ndarray, np.ndarray]:
    mask = table.group_ids == group
    rows = np.asarray([int(float(row["周期编号"])) for row in np.asarray(table.rows, dtype=object)[mask]], dtype=np.int64)
    order = np.argsort(rows)
    return table.features[mask][order], table.target[mask][order]


def _make_windows(
    table: SimulationTable,
    groups: Sequence[str],
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    target_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    if not groups:
        raise ValueError("At least one trajectory group is required")
    trajectories = [_group_rows(table, group) for group in groups]
    if mean is None:
        mean = np.concatenate([features for features, _ in trajectories], axis=0).mean(axis=0)
    if scale is None:
        scale = np.concatenate([features for features, _ in trajectories], axis=0).std(axis=0)
        scale = np.where(scale < 1.0e-8, 1.0, scale)
    if target_scale is None:
        target_scale = max(float(max(target.max() for _, target in trajectories)), 1.0)
    xs, ys, group_values, endpoints = [], [], [], []
    for group, (features, target) in zip(groups, trajectories):
        normalized = (features - mean) / scale
        for endpoint in range(SEQ_LEN - 1, len(features)):
            xs.append(normalized[endpoint - SEQ_LEN + 1 : endpoint + 1])
            ys.append(target[endpoint])
            group_values.append(group)
            endpoints.append(endpoint)
    if not xs:
        raise ValueError("No simulation windows generated")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(group_values),
        float(target_scale),
        np.asarray(endpoints, dtype=np.int64),
        np.asarray(mean, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


def train_model(
    candidate: Candidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    device: str,
    seed: int,
    *,
    epochs: int,
    target_scale: float,
    scheduler_budget: int,
    batch_size: int = 2048,
    patience: int = 25,
) -> tuple[object, int, float]:
    seed_everything(seed)
    model = build_reaction_wheel_model(candidate.model, x_train.shape[-1]).to(device)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train / target_scale, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.0e-4, weight_decay=1.0e-4)
    batches = math.ceil(max(len(x_train), 1) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, max(epochs, scheduler_budget) * batches),
    )
    rng = np.random.default_rng(seed)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device) if x_val is not None else None
    best_state = None
    best_score = float("inf")
    best_epoch = int(epochs)
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            prediction = model(x[index]).rul
            loss = F.smooth_l1_loss(prediction, y[index], beta=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        if x_val is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_prediction = model(val_x).rul.detach().cpu().numpy() * target_scale
        score = rmse(y_val, validation_prediction)
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
        "scheduler_epoch_budget": int(scheduler_budget),
        "batches_per_epoch": int(batches),
    }
    model.eval()
    return model, best_epoch, float(best_score)


@torch.no_grad()
def predict_model(model, x: np.ndarray, target_scale: float, device: str) -> np.ndarray:
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    return (model(tensor).rul.detach().cpu().numpy() * target_scale).astype(np.float32)


def metric_dict(y_true: np.ndarray, prediction: np.ndarray, name: str, *, pre_failure: bool = False) -> dict[str, object]:
    if pre_failure:
        mask = y_true > 0.0
    else:
        mask = np.ones(len(y_true), dtype=bool)
    return {
        "name": name,
        "scope": "pre_failure" if pre_failure else "all_cycles",
        "rmse": rmse(y_true[mask], prediction[mask]),
        "mae": mae(y_true[mask], prediction[mask]),
        "bias": bias(y_true[mask], prediction[mask]),
        "n": int(mask.sum()),
    }


def _regression_matrix(x: np.ndarray, endpoints: np.ndarray, target_scale: float) -> np.ndarray:
    return np.c_[np.ones(len(x)), x[:, -1, :], endpoints / max(target_scale, 1.0)]


def fit_ridge(x: np.ndarray, y: np.ndarray, endpoints: np.ndarray, target_scale: float, alpha: float) -> np.ndarray:
    design = _regression_matrix(x, endpoints, target_scale)
    identity = np.eye(design.shape[1])
    identity[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * identity, design.T @ y)


def predict_ridge(x: np.ndarray, endpoints: np.ndarray, target_scale: float, beta: np.ndarray) -> np.ndarray:
    return np.maximum(_regression_matrix(x, endpoints, target_scale) @ beta, 0.0).astype(np.float32)


def select_candidate(table: SimulationTable, candidates: Sequence[Candidate], device: str, seeds: Sequence[int], epochs: int) -> dict[str, object]:
    train_groups = [group for group in sorted(set(table.group_ids.tolist()), key=lambda value: int(value)) if group in {"1", "2", "3"}]
    validation_groups = ["4"]
    rows = []
    for candidate in candidates:
        train_x, train_y, _, target_scale, train_endpoints, mean, scale = _make_windows(table, train_groups)
        validation_x, validation_y, _, _, validation_endpoints, _, _ = _make_windows(
            table, validation_groups, mean=mean, scale=scale, target_scale=target_scale
        )
        scores, best_epochs = [], []
        for seed in seeds:
            model, best_epoch, score = train_model(
                candidate, train_x, train_y, validation_x, validation_y, device, seed,
                epochs=epochs, target_scale=target_scale, scheduler_budget=epochs,
            )
            scores.append(score)
            best_epochs.append(best_epoch)
        rows.append({
            "candidate": asdict(candidate),
            "validation_rmse": float(np.mean(scores)),
            "validation_seed_rmse": [float(value) for value in scores],
            "median_best_epoch": int(np.median(best_epochs)),
            "n_train": len(train_x),
            "n_validation": len(validation_x),
            "validation_endpoints": [int(validation_endpoints.min()), int(validation_endpoints.max())],
        })
    selected = min(rows, key=lambda row: row["validation_rmse"])
    return {
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "selected": selected["candidate"],
        "fixed_epochs": selected["median_best_epoch"],
        "candidate_rows": rows,
    }


def run_final(table: SimulationTable, selection: Mapping[str, object], device: str, seeds: Sequence[int]) -> dict[str, object]:
    train_groups = ["1", "2", "3", "4"]
    test_groups = ["5"]
    candidate = Candidate(**selection["selected"])
    epochs = int(selection["fixed_epochs"])
    train_x, train_y, _, target_scale, train_endpoints, mean, scale = _make_windows(table, train_groups)
    test_x, test_y, test_bearings, _, test_endpoints, _, _ = _make_windows(
        table, test_groups, mean=mean, scale=scale, target_scale=target_scale
    )
    seed_predictions = []
    seed_metrics = []
    for seed in seeds:
        model, _, _ = train_model(
            candidate, train_x, train_y, None, None, device, seed,
            epochs=epochs, target_scale=target_scale, scheduler_budget=EPOCH_BUDGET,
        )
        prediction = predict_model(model, test_x, target_scale, device)
        seed_predictions.append(prediction)
        seed_metrics.extend([
            metric_dict(test_y, prediction, f"seed_{seed}"),
            metric_dict(test_y, prediction, f"seed_{seed}", pre_failure=True),
        ])
    neural_prediction = np.mean(np.stack(seed_predictions), axis=0)

    # Select Ridge regularization only on the original train/validation split,
    # then refit the frozen alpha on tracks 1-4 before the single test pass.
    selection_x, selection_y, _, selection_scale, selection_endpoints, selection_mean, selection_std = _make_windows(
        table, ["1", "2", "3"]
    )
    validation_x, validation_y, _, _, validation_endpoints, _, _ = _make_windows(
        table, ["4"], mean=selection_mean, scale=selection_std, target_scale=selection_scale
    )
    best_alpha = None
    best_validation = float("inf")
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        beta = fit_ridge(selection_x, selection_y, selection_endpoints, selection_scale, alpha)
        prediction = predict_ridge(validation_x, validation_endpoints, selection_scale, beta)
        score = rmse(validation_y, prediction)
        if score < best_validation:
            best_validation, best_alpha = score, alpha
    beta = fit_ridge(train_x, train_y, train_endpoints, target_scale, float(best_alpha))
    ridge_prediction = predict_ridge(test_x, test_endpoints, target_scale, beta)

    result = {
        "schema": "reaction_wheel_sim_initial_report",
        "domain": "comsol_reaction_wheel_simulation",
        "protocol": "tracks 1-3 train, track 4 validation, track 5 test; operational aligned-cycle features",
        "label": "physical remaining-life cycles from supplied COMSOL aligned table",
        "feature_tier": "operational",
        "seq_len": SEQ_LEN,
        "scheduler_epoch_budget": EPOCH_BUDGET,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "candidate_selection": selection,
        "selected_candidate": asdict(candidate),
        "selected_epochs": epochs,
        "target_scale_train_cycles": target_scale,
        "ridge_alpha_selected_on": "tracks 1-3 train / track 4 validation",
        "ridge_alpha": float(best_alpha),
        "ridge_validation_rmse": float(best_validation),
        "seed_ids": list(seeds),
        "metrics": [
            metric_dict(test_y, neural_prediction, "selected_neural"),
            metric_dict(test_y, neural_prediction, "selected_neural", pre_failure=True),
            metric_dict(test_y, ridge_prediction, "ridge"),
            metric_dict(test_y, ridge_prediction, "ridge", pre_failure=True),
        ],
        "test_rows": [
            {
                "group": str(group),
                "endpoint": int(endpoint),
                "true_rul": float(truth),
                "selected_neural": float(neural),
                "ridge": float(ridge),
            }
            for group, endpoint, truth, neural, ridge in zip(
                test_bearings, test_endpoints, test_y, neural_prediction, ridge_prediction
            )
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="outputs/reaction_wheel_sim_initial")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCH_BUDGET)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    epochs = min(int(args.epochs), 8) if args.quick else int(args.epochs)
    if args.quick:
        seeds = seeds[:1]
    archive = load_simulation_archive(args.archive)
    table = archive.aligned["operational"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    selection = select_candidate(table, CANDIDATES, device, seeds, epochs)
    result = run_final(table, selection, device, seeds)
    result["archive_sha256"] = archive.sha256
    result["elapsed_sec"] = time.time() - started
    write_json(output / "REACTION_WHEEL_SIM_INITIAL_REPORT.json", result)
    write_json(output / "frozen_config.json", {
        "schema": "reaction_wheel_sim_initial_config",
        "archive_sha256": archive.sha256,
        "protocol": result["protocol"],
        "feature_tier": "operational",
        "seq_len": SEQ_LEN,
        "scheduler_epoch_budget": EPOCH_BUDGET,
        "seed_ids": list(seeds),
        "selection": selection,
    })
    print(json.dumps({metric["name"] + ":" + metric["scope"]: metric["rmse"] for metric in result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
