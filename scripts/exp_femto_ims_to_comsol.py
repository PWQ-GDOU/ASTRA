"""Strict FEMTO/IMS -> COMSOL reaction-wheel simulation transfer.

The source bearing datasets are used only for source pretraining.  The target
COMSOL archive keeps its declared trajectory split:

* tracks 1..N: target fine-tuning data (N=1,2,3)
* track 4: chronological validation and early stopping only
* track 5: final test, evaluated once per method/seed ensemble

The target scaler and target label scale are fitted from tracks 1..N only.
FEMTO and IMS are run as separate source arms so their evidence cannot be
silently pooled.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Make the entry point runnable both as ``python scripts/...`` and as a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.femto_strict import load_femto_zip
from src.data.ims_strict import IMSUnit, load_ims_archive, load_ims_processed_dir, sha256_file as ims_sha256_file
from src.data.reaction_wheel_sim import SimulationArchive, SimulationTable, load_simulation_archive
from src.models.cross_domain_transfer import CrossDomainRULModel, copy_shared_encoder, count_parameters


SEEDS = (42, 123, 456, 2026, 3407)
SOURCE_SEQ_LEN = 20
TARGET_SEQ_LEN = 20
SOURCE_EPOCHS = 60
TARGET_EPOCHS = 80
TARGET_N_GROUPS = {
    1: ("1",),
    2: ("1", "2"),
    3: ("1", "2", "3"),
}
TARGET_VALIDATION_GROUPS = ("4",)
TARGET_TEST_GROUPS = ("5",)
TARGET_FEATURE_TIER = "operational"
METHODS = ("transfer", "frozen_encoder", "scratch", "ridge")


@dataclass(frozen=True)
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    units: np.ndarray
    endpoints: np.ndarray
    scale: float
    mean: np.ndarray
    std: np.ndarray
    # Optional causal per-trajectory context derived from the first observed
    # target row; it never contains labels or full-life statistics.
    context: np.ndarray | None = None


@dataclass(frozen=True)
class SourceBundle:
    name: str
    units: Mapping[str, object]
    train_names: tuple[str, ...]
    validation_names: tuple[str, ...]
    train: PreparedData
    validation: PreparedData
    feature_names: tuple[str, ...]
    sha256: str | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


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


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_dir():
        digest = hashlib.sha256()
        for item in sorted((child for child in source.rglob("*") if child.is_file()), key=lambda child: str(child.relative_to(source))):
            digest.update(str(item.relative_to(source)).replace("\\", "/").encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        return digest.hexdigest()
    return ims_sha256_file(source)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if np.any(mask) else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if np.any(mask) else float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(y_pred[mask] - y_true[mask])) if np.any(mask) else float("nan")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float, method: str, seed: int | None = None) -> dict[str, object]:
    raw = rmse(y_true, y_pred)
    return {
        "method": method,
        "seed": seed,
        "raw_rmse": raw,
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "normalized_rmse": float(raw / max(float(scale), 1.0)),
        "n": int(len(y_true)),
    }


def _natural_group(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value


def _fit_stats(units: Sequence[object], feature_count: int) -> tuple[np.ndarray, np.ndarray, float]:
    rows = np.concatenate([np.asarray(item.features, dtype=np.float64) for item in units], axis=0)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std[std < 1.0e-8] = 1.0
    label_scale = max(max(float(np.max(item.raw_rul)) for item in units), 1.0)
    if len(mean) != feature_count:
        raise ValueError("Feature-count mismatch while fitting source statistics")
    return mean.astype(np.float32), std.astype(np.float32), float(label_scale)


def _make_source_windows(
    units: Sequence[object],
    *,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    label_scale: float,
) -> PreparedData:
    if not units:
        raise ValueError("Source window construction requires at least one unit")
    xs: list[np.ndarray] = []
    ys: list[float] = []
    names: list[str] = []
    endpoints: list[int] = []
    for unit in units:
        values = (np.asarray(unit.features, dtype=np.float32) - mean) / std
        for endpoint in range(seq_len - 1, len(unit)):
            xs.append(values[endpoint - seq_len + 1 : endpoint + 1])
            ys.append(float(unit.raw_rul[endpoint]) / label_scale)
            names.append(str(unit.name))
            endpoints.append(endpoint)
    if not xs:
        raise ValueError("Source units are too short for the configured sequence length")
    return PreparedData(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.float32),
        units=np.asarray(names),
        endpoints=np.asarray(endpoints, dtype=np.int64),
        scale=float(label_scale),
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )


def _source_split_names(names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = tuple(sorted(names, key=_natural_group))
    if len(ordered) < 2:
        raise ValueError("At least two source units are required for source-only validation")
    validation_count = max(1, len(ordered) // 3)
    return ordered[:-validation_count], ordered[-validation_count:]


def build_source_bundle(name: str, units: Mapping[str, object], source_path: Path | None, max_files: int | None) -> SourceBundle:
    if name == "femto":
        ordered = {}
        raw = load_femto_zip(source_path, feature_group="full", max_files_per_bearing=max_files)
        ordered.update(raw)
    elif name == "ims":
        if source_path.is_dir():
            ordered = dict(load_ims_processed_dir(source_path))
        else:
            ordered = dict(load_ims_archive(source_path, max_files_per_unit=max_files))
    else:
        raise ValueError(f"Unknown source {name}")
    train_names, validation_names = _source_split_names(tuple(ordered))
    train_units = [ordered[item] for item in train_names]
    validation_units = [ordered[item] for item in validation_names]
    feature_names = tuple(train_units[0].feature_names)
    mean, std, scale = _fit_stats(train_units, len(feature_names))
    train = _make_source_windows(train_units, seq_len=SOURCE_SEQ_LEN, mean=mean, std=std, label_scale=scale)
    validation = _make_source_windows(
        validation_units,
        seq_len=SOURCE_SEQ_LEN,
        mean=mean,
        std=std,
        label_scale=scale,
    )
    return SourceBundle(
        name=name,
        units=ordered,
        train_names=train_names,
        validation_names=validation_names,
        train=train,
        validation=validation,
        feature_names=feature_names,
        sha256=sha256_file(source_path) if source_path is not None and source_path.exists() else None,
    )


def _group_rows(table: SimulationTable, group: str) -> tuple[np.ndarray, np.ndarray]:
    mask = table.group_ids == group
    rows = np.asarray(
        [int(float(row["周期编号"])) for row in np.asarray(table.rows, dtype=object)[mask]],
        dtype=np.int64,
    )
    order = np.argsort(rows)
    return table.features[mask][order], table.target[mask][order]


def _target_stats(table: SimulationTable, groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray, float]:
    trajectories = [_group_rows(table, group) for group in groups]
    if not trajectories:
        raise ValueError("Target statistics require at least one training trajectory")
    rows = np.concatenate([features for features, _ in trajectories], axis=0).astype(np.float64)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std[std < 1.0e-8] = 1.0
    label_scale = max(max(float(np.max(target)) for _, target in trajectories), 1.0)
    return mean.astype(np.float32), std.astype(np.float32), float(label_scale)


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
    for group in groups:
        features, target = _group_rows(table, group)
        values = (features.astype(np.float32) - mean) / std
        for endpoint in range(TARGET_SEQ_LEN - 1, len(values)):
            xs.append(values[endpoint - TARGET_SEQ_LEN + 1 : endpoint + 1])
            ys.append(float(target[endpoint]) / label_scale)
            names.append(group)
            endpoints.append(endpoint)
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
    )


def prepare_target(table: SimulationTable, train_groups: Sequence[str]) -> tuple[PreparedData, PreparedData, PreparedData, dict[str, object]]:
    mean, std, label_scale = _target_stats(table, train_groups)
    train = _make_target_windows(table, train_groups, mean=mean, std=std, label_scale=label_scale)
    validation = _make_target_windows(table, TARGET_VALIDATION_GROUPS, mean=mean, std=std, label_scale=label_scale)
    test = _make_target_windows(table, TARGET_TEST_GROUPS, mean=mean, std=std, label_scale=label_scale)
    audit = {
        "fit_groups": list(train_groups),
        "validation_groups": list(TARGET_VALIDATION_GROUPS),
        "test_groups": list(TARGET_TEST_GROUPS),
        "feature_names": list(table.feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "label_scale_from_fit_groups": label_scale,
        "test_labels_used_for_fit": False,
        "test_labels_used_for_validation": False,
    }
    return train, validation, test, audit


def _best_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_loop(
    model: CrossDomainRULModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int,
    source: bool,
    frozen_encoder: bool = False,
    patience: int = 15,
    sample_weights: np.ndarray | None = None,
) -> tuple[CrossDomainRULModel, dict[str, object]]:
    seed_everything(seed)
    model = model.to(device)
    if frozen_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=7.0e-4, weight_decay=1.0e-4)
    batch_size = min(2048, max(16, len(x_train)))
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = int(epochs)
    best_state = None
    stale = 0
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    if sample_weights is None:
        train_weights = None
    else:
        weights = np.asarray(sample_weights, dtype=np.float32)
        if weights.shape != (len(x_train),):
            raise ValueError("sample_weights must have one finite positive value per training sample")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("sample_weights must have one finite positive value per training sample")
        train_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    validation_y = np.asarray(y_validation, dtype=np.float32)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            output = model.forward_source(train_x[index]) if source else model.forward_target(train_x[index])
            residual_loss = F.smooth_l1_loss(
                output.rul, train_y[index], beta=0.05, reduction="none"
            )
            if train_weights is None:
                loss = residual_loss.mean()
            else:
                batch_weights = train_weights[index]
                loss = torch.sum(residual_loss * batch_weights) / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model.forward_source(validation_x) if source else model.forward_target(validation_x)
            prediction = output.rul.detach().cpu().numpy()
        score = rmse(validation_y, prediction)
        if score < best_score - 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = _best_state(model)
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
        "frozen_encoder": bool(frozen_encoder),
    }


def _new_seeded_model(source_features: int, target_features: int, *, seed: int) -> CrossDomainRULModel:
    """Construct after seeding so initialization is part of the protocol."""
    seed_everything(seed)
    return CrossDomainRULModel(source_features, target_features)


def train_source_model(bundle: SourceBundle, *, target_features: int, seed: int, epochs: int, device: str) -> tuple[CrossDomainRULModel, dict[str, object]]:
    model = _new_seeded_model(len(bundle.feature_names), target_features, seed=seed)
    model, diagnostics = _train_loop(
        model,
        bundle.train.x,
        bundle.train.y,
        bundle.validation.x,
        bundle.validation.y,
        device=device,
        seed=seed,
        epochs=epochs,
        source=True,
    )
    diagnostics["model_initialization_seed"] = int(seed)
    return model, diagnostics


def _new_target_model(
    source_model: CrossDomainRULModel | None,
    source_features: int,
    target_features: int,
    method: str,
    device: str,
    initialization_seed: int,
) -> CrossDomainRULModel:
    model = _new_seeded_model(source_features, target_features, seed=initialization_seed)
    if method in {"transfer", "frozen_encoder"}:
        if source_model is None:
            raise ValueError(f"{method} requires a source model")
        copy_shared_encoder(source_model, model)
    if method == "frozen_encoder":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    return model.to(device)


@torch.no_grad()
def _predict_target(model: CrossDomainRULModel, data: PreparedData, *, device: str) -> np.ndarray:
    tensor = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    return (model.forward_target(tensor).rul.detach().cpu().numpy() * data.scale).astype(np.float32)


def _ridge_design(data: PreparedData) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(data.x), dtype=np.float64),
            data.x[:, -1, :].astype(np.float64),
            data.endpoints.astype(np.float64) / max(float(data.scale), 1.0),
        ]
    )


def _fit_ridge(train: PreparedData, alpha: float) -> np.ndarray:
    design = _ridge_design(train)
    identity = np.eye(design.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + float(alpha) * identity, design.T @ train.y)


def _predict_ridge(data: PreparedData, beta: np.ndarray) -> np.ndarray:
    return np.maximum(_ridge_design(data) @ beta, 0.0).astype(np.float32) * data.scale


def _select_ridge_alpha(train: PreparedData, validation: PreparedData) -> tuple[float, float]:
    best_alpha, best_score = None, float("inf")
    for alpha in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0):
        beta = _fit_ridge(train, alpha)
        score = rmse(validation.y * validation.scale, _predict_ridge(validation, beta))
        if score < best_score:
            best_alpha, best_score = alpha, score
    return float(best_alpha), float(best_score)


def run_target_arm(
    source_name: str,
    source_bundle: SourceBundle,
    source_models: Mapping[int, CrossDomainRULModel],
    table: SimulationTable,
    *,
    seeds: Sequence[int],
    epochs: int,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    n_audit: dict[str, object] = {}
    target_features = len(table.feature_names)
    for n_shots, train_groups in TARGET_N_GROUPS.items():
        train, validation, test, audit = prepare_target(table, train_groups)
        n_audit[str(n_shots)] = audit
        method_predictions: dict[str, list[np.ndarray]] = {method: [] for method in METHODS if method != "ridge"}
        for method in ("transfer", "frozen_encoder", "scratch"):
            for seed in seeds:
                source_model = source_models.get(seed)
                training_seed = seed + n_shots * 1000
                model = _new_target_model(
                    source_model,
                    len(source_bundle.feature_names),
                    target_features,
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
                fit_diag["model_initialization_seed"] = int(training_seed)
                prediction = _predict_target(model, test, device=device)
                method_predictions[method].append(prediction)
                metric = metrics(test.y * test.scale, prediction, test.scale, method, seed)
                rows.append(
                    {
                        "source": source_name,
                        "n_shots": n_shots,
                        "method": method,
                        **metric,
                        "fit_groups": list(train_groups),
                        "validation_groups": list(TARGET_VALIDATION_GROUPS),
                        "test_groups": list(TARGET_TEST_GROUPS),
                        "fit_diagnostics": fit_diag,
                    }
                )
        ridge_alpha, ridge_validation_rmse = _select_ridge_alpha(train, validation)
        ridge_train = train
        beta = _fit_ridge(ridge_train, ridge_alpha)
        ridge_prediction = _predict_ridge(test, beta)
        method_predictions["ridge"] = [ridge_prediction]
        rows.append(
            {
                "source": source_name,
                "n_shots": n_shots,
                "method": "ridge",
                **metrics(test.y * test.scale, ridge_prediction, test.scale, "ridge", None),
                "fit_groups": list(train_groups),
                "validation_groups": list(TARGET_VALIDATION_GROUPS),
                "test_groups": list(TARGET_TEST_GROUPS),
                "ridge_alpha": ridge_alpha,
                "ridge_validation_rmse": ridge_validation_rmse,
            }
        )
        truth = test.y * test.scale
        for method, predictions in method_predictions.items():
            aggregate = np.mean(np.stack(predictions), axis=0)
            aggregate_metrics = metrics(truth, aggregate, test.scale, method, None)
            aggregate_metrics["method"] = f"{method}_ensemble"
            rows.append(
                {
                    "source": source_name,
                    "n_shots": n_shots,
                    "method": f"{method}_ensemble",
                    **aggregate_metrics,
                    "fit_groups": list(train_groups),
                    "validation_groups": list(TARGET_VALIDATION_GROUPS),
                    "test_groups": list(TARGET_TEST_GROUPS),
                }
            )
            for endpoint, actual, predicted in zip(test.endpoints, truth, aggregate):
                prediction_rows.append(
                    {
                        "source": source_name,
                        "n_shots": n_shots,
                        "method": f"{method}_ensemble",
                        "target_group": "5",
                        "endpoint": int(endpoint),
                        "true_rul": float(actual),
                        "prediction": float(predicted),
                    }
                )
    return rows, prediction_rows, n_audit


def _source_manifest(bundle: SourceBundle) -> dict[str, object]:
    return {
        "source": bundle.name,
        "sha256": bundle.sha256,
        "unit_names": sorted(bundle.units),
        "source_train_units": list(bundle.train_names),
        "source_validation_units": list(bundle.validation_names),
        "feature_names": list(bundle.feature_names),
        "train_only_scaler_units": list(bundle.train_names),
        "source_label_scale": bundle.train.scale,
        "source_train_windows": len(bundle.train.x),
        "source_validation_windows": len(bundle.validation.x),
        "target_data_used_for_source_fit": False,
    }


def run(source_names: Sequence[str], args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if args.quick:
        seeds = seeds[:2]
    source_epochs = min(args.source_epochs, 8) if args.quick else args.source_epochs
    target_epochs = min(args.target_epochs, 8) if args.quick else args.target_epochs
    max_files = args.source_max_files
    if args.quick and max_files is None:
        max_files = 120

    comsol_path = Path(args.comsol_archive)
    source_paths = {
        "femto": Path(args.femto_zip),
        "ims": Path(args.ims_path),
    }
    missing = [str(path) for path in [comsol_path, *[source_paths[name] for name in source_names]] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    archive: SimulationArchive = load_simulation_archive(comsol_path)
    table = archive.aligned[TARGET_FEATURE_TIER]
    groups = sorted(set(table.group_ids.tolist()), key=_natural_group)
    if groups != ["1", "2", "3", "4", "5"]:
        raise ValueError(f"COMSOL strict protocol requires groups 1..5, found {groups}")
    if len(set(table.group_ids.tolist())) != 5:
        raise ValueError("COMSOL trajectory IDs are not unique")

    all_rows: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    source_reports: dict[str, object] = {}
    for source_name in source_names:
        bundle = build_source_bundle(source_name, {}, source_paths[source_name], max_files)
        source_models: dict[int, CrossDomainRULModel] = {}
        source_training: list[dict[str, object]] = []
        for seed in seeds:
            model, diagnostic = train_source_model(
                bundle,
                target_features=len(table.feature_names),
                seed=seed,
                epochs=source_epochs,
                device=device,
            )
            source_models[seed] = copy.deepcopy(model).cpu()
            source_training.append({"seed": seed, **diagnostic})
        rows, prediction_rows, target_audit = run_target_arm(
            source_name,
            bundle,
            source_models,
            table,
            seeds=seeds,
            epochs=target_epochs,
            device=device,
        )
        all_rows.extend(rows)
        all_predictions.extend(prediction_rows)
        source_reports[source_name] = {
            "manifest": _source_manifest(bundle),
            "source_training": source_training,
            "target_audit_by_n": target_audit,
            "source_model_parameter_count": count_parameters(source_models[seeds[0]]),
        }

    preflight = {
        "schema": "femto_ims_to_comsol_preflight_v1",
        "comsol_archive": str(comsol_path),
        "comsol_sha256": archive.sha256,
        "comsol_feature_tier": TARGET_FEATURE_TIER,
        "comsol_groups": groups,
        "comsol_train_groups": ["1", "2", "3"],
        "comsol_validation_groups": list(TARGET_VALIDATION_GROUPS),
        "comsol_test_groups": list(TARGET_TEST_GROUPS),
        "target_test_labels_used_for_fit": False,
        "target_test_labels_used_for_validation": False,
        "source_names": list(source_names),
        "source_manifests": {name: source_reports[name]["manifest"] for name in source_names},
        "adapter_shape": {
            "source_feature_counts": {
                name: len(source_reports[name]["manifest"]["feature_names"]) for name in source_names
            },
            "target_feature_count": len(table.feature_names),
            "separate_source_target_projection": True,
        },
        "finite_checks": all(np.all(np.isfinite(table.features)) for _ in [0]),
    }
    write_json(output / "PREFLIGHT.json", preflight)
    data_manifest = {
        "schema": "femto_ims_to_comsol_data_manifest_v1",
        "inputs": {
            "comsol": {
                "path": str(comsol_path),
                "sha256": archive.sha256,
                "feature_tier": TARGET_FEATURE_TIER,
                "trajectory_groups": groups,
            },
            "femto": {
                "path": str(source_paths["femto"]),
                "sha256": source_reports.get("femto", {}).get("manifest", {}).get("sha256"),
                "units": source_reports.get("femto", {}).get("manifest", {}).get("unit_names", []),
            },
            "ims": {
                "path": str(source_paths["ims"]),
                "sha256": source_reports.get("ims", {}).get("manifest", {}).get("sha256"),
                "units": source_reports.get("ims", {}).get("manifest", {}).get("unit_names", []),
                "label_semantics": "processed public IMS source runs; 2nd_test train and 3rd_test validation",
            },
        },
        "target_split": {
            "fine_tune_by_n": {str(key): list(value) for key, value in TARGET_N_GROUPS.items()},
            "validation": list(TARGET_VALIDATION_GROUPS),
            "test": list(TARGET_TEST_GROUPS),
            "test_labels_used_for_selection": False,
        },
    }
    write_json(output / "DATA_MANIFEST.json", data_manifest)
    protocol = {
        "schema": "femto_ims_to_comsol_transfer_v1",
        "source_arms": list(source_names),
        "source_protocol": "source-unit-disjoint pretraining; source validation only; no COMSOL rows or labels",
        "target_protocol": "COMSOL operational aligned-cycle features; tracks 1..N fine-tune, track 4 chronological validation, track 5 final test",
        "target_shots": {str(key): list(value) for key, value in TARGET_N_GROUPS.items()},
        "target_feature_tier": TARGET_FEATURE_TIER,
        "target_scaler": "fit on target fine-tuning tracks only",
        "target_label_scale": "fit on target fine-tuning tracks only",
        "model": "source/target domain-specific projections + shared GRU encoder + independently initialized target head",
        "transfer": "pretrained shared encoder copied; target projection and target head initialized independently",
        "frozen_ablation": "shared encoder copied and frozen; target projection/head trainable",
        "scratch": "same target architecture, data, seeds, epochs and early stopping with random initialization",
        "ridge": "target-only baseline; alpha selected on track 4 and refit on tracks 1..N before track 5",
        "seeds": list(seeds),
        "randomness_control": "Each source and target model is constructed only after its explicit seed is set; training resets the same seed, deterministic algorithms are requested, and CPU MKLDNN is disabled.",
        "source_epochs": source_epochs,
        "target_epochs": target_epochs,
        "test_evaluated_once_after_selection": True,
        "proxy_warning": "COMSOL is a reaction-wheel simulation domain, not real flight telemetry",
    }
    write_json(output / "PROTOCOL.json", protocol)
    write_json(
        output / "REPORT.json",
        {
            "schema": "femto_ims_to_comsol_transfer_report_v1",
            "protocol": protocol,
            "preflight": preflight,
            "sources": source_reports,
            "macro_rows": all_rows,
            "elapsed_sec": time.time() - started,
        },
    )
    with (output / "MACRO.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(all_rows))
    with (output / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["source", "n_shots", "method", "target_group", "endpoint", "true_rul", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_predictions)
    return {"output": str(output), "sources": list(source_names), "rows": len(all_rows), "elapsed_sec": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("femto", "ims", "both"), default="both")
    parser.add_argument("--femto-zip", default="data/processed/femto_bearing.zip")
    parser.add_argument("--ims-path", default="data/processed/ims_processed")
    parser.add_argument("--comsol-archive", default="data/raw/competition/reaction_wheel_comsol_degradation.zip")
    parser.add_argument("--output", default="outputs/femto_ims_to_comsol_transfer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--source-epochs", type=int, default=SOURCE_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=TARGET_EPOCHS)
    parser.add_argument("--source-max-files", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    source_names = ("femto", "ims") if args.source == "both" else (args.source,)
    result = run(source_names, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
