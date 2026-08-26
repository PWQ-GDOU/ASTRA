"""Auditable FEMTO/IMS -> COMSOL outer-fold transfer with trend residuals.

The v4 target protocol keeps a causal target-side trend as an explicit prior
and transfers only the residual dynamics.  The prior is fitted from the
fine-tuning trajectories and selected on the chronological validation
trajectory.  The outer holdout is never used for scaling, prior fitting,
model selection, or early stopping.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_femto_ims_to_comsol import (  # noqa: E402
    SEEDS,
    SOURCE_EPOCHS,
    TARGET_EPOCHS,
    TARGET_FEATURE_TIER,
    TARGET_SEQ_LEN,
    PreparedData,
    _json_safe,
    _fit_ridge,
    _natural_group,
    _predict_ridge,
    _select_ridge_alpha,
    _target_stats,
    build_source_bundle,
    count_parameters,
    metrics,
    rmse,
    seed_everything,
    sha256_file,
    _new_seeded_model,
    _train_loop,
    write_json,
)
from scripts.exp_femto_ims_to_comsol_outerloo import (  # noqa: E402
    _fit_huber_trend,
    _fit_trend,
    _predict_trend,
    _make_target_windows,
    outer_schedule,
    prepare_outer_target,
)
from src.data.reaction_wheel_sim import SimulationArchive, SimulationTable, load_simulation_archive  # noqa: E402
from src.models.cross_domain_transfer import (  # noqa: E402
    CrossDomainRULModel,
    TrendResidualRULModel,
    copy_shared_encoder_to_trend_residual,
)


TARGET_N_SHOTS = (1, 2, 3)
TREND_METHODS = ("trend_linear", "trend_quadratic", "trend_huber")
CALIBRATED_TRANSFER_METHOD = "trend_residual_transfer_calibrated"
RIDGE_CALIBRATED_TRANSFER_METHOD = "ridge_calibrated_transfer"
AFFINE_CALIBRATED_TRANSFER_METHOD = "trend_residual_transfer_affine_calibrated"
CALIBRATED_SCRATCH_METHOD = "trend_residual_scratch_calibrated"
RIDGE_CALIBRATED_SCRATCH_METHOD = "ridge_calibrated_scratch"
AFFINE_CALIBRATED_SCRATCH_METHOD = "trend_residual_scratch_affine_calibrated"
CALIBRATED_TREND_FAMILY = "trend_huber"
CALIBRATED_DEFAULT_TRANSFER_WEIGHT = 0.5
CALIBRATED_WEIGHT_POLICIES = ("fixed", "validation", "fit_only_innerloo")
BLEND_SELECTION_POLICIES = ("inner_group_loo", "validation")
CALIBRATED_MIN_TRANSFER_WEIGHT = 0.25
AFFINE_SLOPE_BOUNDS = (0.5, 1.5)
AFFINE_INTERCEPT_FRACTION = 0.25
DEFAULT_TRANSFER_WARMUP_EPOCHS = 2
PRIOR_POLICIES = ("validation", "ridge", "linear", "quadratic", "huber")
METHODS = (
    "trend_residual_transfer",
    CALIBRATED_TRANSFER_METHOD,
    AFFINE_CALIBRATED_TRANSFER_METHOD,
    RIDGE_CALIBRATED_TRANSFER_METHOD,
    "trend_residual_frozen",
    "trend_residual_scratch",
    CALIBRATED_SCRATCH_METHOD,
    AFFINE_CALIBRATED_SCRATCH_METHOD,
    RIDGE_CALIBRATED_SCRATCH_METHOD,
    "ridge",
    *TREND_METHODS,
)
SEED_METHODS = ("trend_residual_transfer", "trend_residual_frozen", "trend_residual_scratch")
MATCHED_SCRATCH_REFERENCE = {
    "trend_residual_transfer_ensemble": "trend_residual_scratch_ensemble",
    f"{CALIBRATED_TRANSFER_METHOD}_ensemble": f"{CALIBRATED_SCRATCH_METHOD}_ensemble",
    f"{AFFINE_CALIBRATED_TRANSFER_METHOD}_ensemble": f"{AFFINE_CALIBRATED_SCRATCH_METHOD}_ensemble",
    f"{RIDGE_CALIBRATED_TRANSFER_METHOD}_ensemble": f"{RIDGE_CALIBRATED_SCRATCH_METHOD}_ensemble",
}
TRANSFER_EVIDENCE_METHODS = frozenset(MATCHED_SCRATCH_REFERENCE)


def _fold_number(value: str) -> int:
    return _natural_group(str(value))[0]


def _make_prior_data(data: PreparedData, prior_normalized: np.ndarray) -> PreparedData:
    prior = np.asarray(prior_normalized, dtype=np.float32)
    if len(prior) != len(data.y) or not np.all(np.isfinite(prior)):
        raise ValueError("Trend prior shape or finiteness check failed")
    return PreparedData(
        x=data.x,
        y=(data.y - prior).astype(np.float32),
        units=data.units,
        endpoints=data.endpoints,
        scale=data.scale,
        mean=data.mean,
        std=data.std,
        context=None if data.context is None else data.context.copy(),
    )


def _append_target_deltas(data: PreparedData) -> PreparedData:
    """Append causal first differences to the target window features."""
    if data.x.ndim != 3 or data.x.shape[1] < 1:
        raise ValueError("Target windows must be a non-empty rank-3 tensor")
    delta = np.zeros_like(data.x, dtype=np.float32)
    if data.x.shape[1] > 1:
        delta[:, 1:, :] = data.x[:, 1:, :] - data.x[:, :-1, :]
    features = np.concatenate([data.x, delta], axis=2).astype(np.float32)
    return PreparedData(
        x=features,
        y=data.y.copy(),
        units=data.units.copy(),
        endpoints=data.endpoints.copy(),
        scale=data.scale,
        mean=data.mean.copy(),
        std=data.std.copy(),
        context=None if data.context is None else data.context.copy(),
    )


def _source_unit_weights(units: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Give every source trajectory equal total loss mass during pretraining."""
    values = np.asarray(units, dtype=str)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Source units must be a non-empty rank-1 array")
    names, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(names) < 2 or np.any(counts <= 0):
        raise ValueError("Source unit weighting requires at least two non-empty units")
    weights = 1.0 / counts[inverse].astype(np.float32)
    weights /= float(np.mean(weights))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise FloatingPointError("Source unit weighting produced invalid weights")
    effective_mass = {
        str(name): float(weights[inverse == index].sum())
        for index, name in enumerate(names)
    }
    return weights.astype(np.float32), {
        "policy": "unit_equal_weighted_loss",
        "unit_window_counts": {str(name): int(count) for name, count in zip(names, counts)},
        "unit_effective_loss_mass": effective_mass,
        "mean_sample_weight": float(weights.mean()),
        "target_data_used": False,
    }


def _train_source_residual_model(
    bundle,
    *,
    target_features: int,
    seed: int,
    epochs: int,
    device: str,
) -> tuple[CrossDomainRULModel, dict[str, object]]:
    """Pretrain the shared encoder on source trend residuals.

    The source prior is fitted from source training trajectories only.  Source
    validation is used for early stopping and prior-family selection, matching
    the target residual objective without using any COMSOL information.
    """
    priors, prior_diag = _select_prior(bundle.train, bundle.validation, bundle.validation)
    train_residual = _make_prior_data(bundle.train, priors["train"])
    validation_residual = _make_prior_data(bundle.validation, priors["validation"])
    train_weights, weighting_audit = _source_unit_weights(bundle.train.units)
    model = _new_seeded_model(len(bundle.feature_names), target_features, seed=seed)
    model, diagnostics = _train_loop(
        model,
        train_residual.x,
        train_residual.y,
        validation_residual.x,
        validation_residual.y,
        device=device,
        seed=seed,
        epochs=epochs,
        source=True,
        sample_weights=train_weights,
    )
    diagnostics.update({
        "model_initialization_seed": int(seed),
        "source_residual_pretraining": True,
        "source_trend_prior": prior_diag,
        "source_prior_fit_groups": list(bundle.train_names),
        "source_validation_used_for_early_stopping": True,
        "source_sampling": weighting_audit,
    })
    return model, diagnostics


def _align_target_adapter_to_source(
    source_model: CrossDomainRULModel,
    target_model: TrendResidualRULModel,
    source_train: PreparedData,
    target_train: PreparedData,
    *,
    device: str,
    seed: int,
    steps: int,
) -> dict[str, object]:
    """Align a target input adapter using source and target fit inputs only.

    The shared source encoder stays frozen while the target projection matches
    source projected/encoded feature moments. This unsupervised initialization
    makes the transferred encoder usable before low-shot supervised fine-tuning
    without reading target validation or outer-holdout samples or labels.
    """
    if int(steps) < 0:
        raise ValueError("Target adapter alignment steps must be non-negative")
    if int(steps) == 0:
        return {
            "enabled": False,
            "steps": 0,
            "source_samples": 0,
            "target_fit_samples": 0,
            "target_labels_used": False,
            "validation_or_holdout_used": False,
        }
    if len(source_train.x) < 1 or len(target_train.x) < 1:
        raise ValueError("Target adapter alignment requires non-empty source and target fit windows")

    seed_everything(seed)
    source_model = source_model.to(device)
    target_model = target_model.to(device)
    source_model.eval()
    source_x = torch.as_tensor(source_train.x, dtype=torch.float32, device=device)
    target_x = torch.as_tensor(target_train.x, dtype=torch.float32, device=device)
    if source_x.ndim != 3 or target_x.ndim != 3:
        raise ValueError("Target adapter alignment expects rank-3 source and target window tensors")
    original_encoder_requires_grad = [parameter.requires_grad for parameter in target_model.encoder.parameters()]
    for parameter in target_model.encoder.parameters():
        parameter.requires_grad = False
    # cuDNN needs a training-mode GRU for gradients to flow from the frozen
    # encoder output back into the trainable target projection.
    target_model.encoder.train()
    optimizer = torch.optim.AdamW(target_model.target_projection.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(seed)
    batch_size = min(512, len(source_x), len(target_x))
    losses: list[float] = []
    try:
        target_model.target_projection.train()
        for _ in range(int(steps)):
            source_index = torch.as_tensor(
                rng.integers(0, len(source_x), size=batch_size), dtype=torch.long, device=device
            )
            target_index = torch.as_tensor(
                rng.integers(0, len(target_x), size=batch_size), dtype=torch.long, device=device
            )
            with torch.no_grad():
                source_projected = source_model.source_projection(source_x[source_index])
                source_encoded, _ = source_model.encoder(source_projected)
            target_projected = target_model.target_projection(target_x[target_index])
            target_encoded, _ = target_model.encoder(target_projected)
            source_projection_flat = source_projected.reshape(-1, source_projected.shape[-1])
            target_projection_flat = target_projected.reshape(-1, target_projected.shape[-1])
            projection_loss = F.mse_loss(
                target_projection_flat.mean(dim=0), source_projection_flat.mean(dim=0)
            ) + F.mse_loss(
                target_projection_flat.var(dim=0, unbiased=False), source_projection_flat.var(dim=0, unbiased=False)
            )
            hidden_loss = F.mse_loss(
                target_encoded[:, -1].mean(dim=0), source_encoded[:, -1].mean(dim=0)
            ) + F.mse_loss(
                target_encoded[:, -1].var(dim=0, unbiased=False), source_encoded[:, -1].var(dim=0, unbiased=False)
            )
            loss = projection_loss + hidden_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Target adapter alignment produced non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.target_projection.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    finally:
        for parameter, required in zip(target_model.encoder.parameters(), original_encoder_requires_grad):
            parameter.requires_grad = required
        target_model.eval()
    return {
        "enabled": True,
        "steps": int(steps),
        "source_samples": int(len(source_train.x)),
        "target_fit_samples": int(len(target_train.x)),
        "objective": "source_target_projected_and_encoded_moment_matching",
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "target_labels_used": False,
        "validation_or_holdout_used": False,
    }


def _trend_normalized(beta: np.ndarray, data: PreparedData, degree: int, endpoint_scale: float) -> np.ndarray:
    raw = _predict_trend(beta, data, degree=degree, endpoint_scale=endpoint_scale)
    return (raw / max(float(data.scale), 1.0)).astype(np.float32)


def _fit_linear_prior(train: PreparedData, validation: PreparedData) -> tuple[str, np.ndarray, dict[str, object]]:
    endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
    beta = _fit_trend(train, degree=1, endpoint_scale=endpoint_scale)
    validation_prediction = _trend_normalized(beta, validation, 1, endpoint_scale)
    return "trend_linear", beta, {
        "degree": 1,
        "endpoint_scale_from_fit_groups": endpoint_scale,
        "validation_raw_rmse": rmse(validation.y * validation.scale, validation_prediction * validation.scale),
    }


def _fit_quadratic_prior(train: PreparedData, validation: PreparedData) -> tuple[str, np.ndarray, dict[str, object]]:
    endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
    best: tuple[float, np.ndarray, float] | None = None
    for alpha in (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0):
        beta = _fit_trend(train, degree=2, endpoint_scale=endpoint_scale, alpha=alpha)
        prediction = _trend_normalized(beta, validation, 2, endpoint_scale)
        score = rmse(validation.y * validation.scale, prediction * validation.scale)
        if best is None or score < best[2]:
            best = (alpha, beta, score)
    if best is None:
        raise RuntimeError("No quadratic trend candidate")
    alpha, beta, score = best
    return "trend_quadratic", beta, {
        "degree": 2,
        "selected_alpha": float(alpha),
        "endpoint_scale_from_fit_groups": endpoint_scale,
        "validation_raw_rmse": float(score),
    }


def _fit_huber_prior(train: PreparedData, validation: PreparedData) -> tuple[str, np.ndarray, dict[str, object]]:
    endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
    best: tuple[float, np.ndarray, float] | None = None
    for delta in (0.75, 1.0, 1.35, 2.0):
        beta = _fit_huber_trend(train, endpoint_scale, delta)
        prediction = _trend_normalized(beta, validation, 1, endpoint_scale)
        score = rmse(validation.y * validation.scale, prediction * validation.scale)
        if best is None or score < best[2]:
            best = (delta, beta, score)
    if best is None:
        raise RuntimeError("No Huber trend candidate")
    delta, beta, score = best
    return "trend_huber", beta, {
        "degree": 1,
        "selected_delta": float(delta),
        "endpoint_scale_from_fit_groups": endpoint_scale,
        "validation_raw_rmse": float(score),
    }


def _fit_ridge_prior(train: PreparedData, validation: PreparedData) -> tuple[str, np.ndarray, dict[str, object]]:
    """Fit a train-only target linear prior for residual transfer.

    This is the same Ridge family reported as a baseline, but when selected as
    a prior the transfer model learns a residual correction on top of it. The
    alpha is chosen on the chronological validation trajectory only.
    """
    alpha, validation_score = _select_ridge_alpha(train, validation)
    beta = _fit_ridge(train, alpha)
    return "ridge", beta, {
        "family": "causal_target_ridge_prior",
        "alpha": float(alpha),
        "validation_raw_rmse": float(validation_score),
    }


def _select_prior(
    train: PreparedData,
    validation: PreparedData,
    test: PreparedData,
    *,
    policy: str = "validation",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if policy not in PRIOR_POLICIES:
        raise ValueError(f"Unknown target prior policy: {policy}")
    candidates = [
        _fit_linear_prior(train, validation),
        _fit_quadratic_prior(train, validation),
        _fit_huber_prior(train, validation),
        _fit_ridge_prior(train, validation),
    ]
    policy_method = {
        "ridge": "ridge",
        "linear": "trend_linear",
        "quadratic": "trend_quadratic",
        "huber": "trend_huber",
    }
    if policy == "validation":
        selected_name, selected_beta, selected_diag = min(
            candidates,
            key=lambda item: float(item[2]["validation_raw_rmse"]),
        )
    else:
        selected_name, selected_beta, selected_diag = next(
            item for item in candidates if item[0] == policy_method[policy]
        )
    # Keep the complete candidate diagnostics, while materializing only the
    # selected prior for model input. Ridge is already returned in raw units;
    # the trend families are represented in normalized label units.
    if selected_name == "ridge":
        selected = (_predict_ridge(train, selected_beta) / max(float(train.scale), 1.0)).astype(np.float32)
        selected_validation = (_predict_ridge(validation, selected_beta) / max(float(validation.scale), 1.0)).astype(np.float32)
        selected_test = (_predict_ridge(test, selected_beta) / max(float(test.scale), 1.0)).astype(np.float32)
    else:
        degree = int(selected_diag["degree"])
        endpoint_scale = float(selected_diag["endpoint_scale_from_fit_groups"])
        selected = _trend_normalized(selected_beta, train, degree, endpoint_scale)
        selected_validation = _trend_normalized(selected_beta, validation, degree, endpoint_scale)
        selected_test = _trend_normalized(selected_beta, test, degree, endpoint_scale)
    return {
        "selected": np.concatenate([selected, selected_validation, selected_test]),
        "train": selected,
        "validation": selected_validation,
        "test": selected_test,
    }, {
        "selected_method": selected_name,
        "selection_policy": policy,
        "selected_beta": selected_beta.tolist(),
        "selected_diagnostics": selected_diag,
        "candidates": {
            name: {**diag, "beta": beta.tolist()}
            for name, beta, diag in candidates
        },
        "full_life_target_labels_used_for_prior": False,
        "holdout_labels_used_for_prior": False,
    }


def _new_model(
    target_features: int,
    seed: int,
    *,
    context_features: int = 0,
) -> TrendResidualRULModel:
    seed_everything(seed)
    return TrendResidualRULModel(
        target_features,
        context_features=(context_features or None),
    )


def _best_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_residual_model(
    model: TrendResidualRULModel,
    train: PreparedData,
    validation: PreparedData,
    prior_train: np.ndarray,
    prior_validation: np.ndarray,
    *,
    device: str,
    seed: int,
    epochs: int,
    frozen_encoder: bool = False,
    warmup_frozen_epochs: int = 0,
    require_unfrozen_checkpoint: bool = False,
    encoder_anchor: dict[str, torch.Tensor] | None = None,
    encoder_anchor_weight: float = 0.0,
    patience: int = 15,
) -> tuple[TrendResidualRULModel, dict[str, object]]:
    """Fit a residual model with optional source-preserving adaptation.

    Transfer starts with a short projection/head-only warmup, then unfreezes
    the encoder at a lower learning rate while applying an anchor penalty to
    the pretrained encoder. The validation trajectory decides which epoch is
    retained; the outer holdout is never read during this process.
    """
    seed_everything(seed)
    model = model.to(device)
    warmup = 0 if frozen_encoder else min(max(0, int(warmup_frozen_epochs)), max(0, int(epochs) - 1))
    if require_unfrozen_checkpoint and (frozen_encoder or warmup == 0):
        raise ValueError("An unfrozen checkpoint requires a non-frozen model with a positive warmup")

    def set_encoder_trainable(enabled: bool) -> None:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = bool(enabled)

    set_encoder_trainable(not frozen_encoder and warmup == 0)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=7.0e-4, weight_decay=1.0e-4)
    batch_size = min(2048, max(16, len(train.x)))
    rng = np.random.default_rng(seed)
    train_x = torch.as_tensor(train.x, dtype=torch.float32, device=device)
    train_prior = torch.as_tensor(prior_train, dtype=torch.float32, device=device)
    train_context = torch.as_tensor(
        train.context if train.context is not None else np.zeros((len(train.x), model.context_features), dtype=np.float32),
        dtype=torch.float32, device=device,
    )
    # The model output is the complete normalized RUL ``prior + residual``.
    # Compare it with the complete target, not ``target + prior``; the latter
    # double-counts the causal prior and makes residual adaptation optimize the
    # wrong objective.
    train_y = torch.as_tensor(train.y, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(validation.x, dtype=torch.float32, device=device)
    validation_prior = torch.as_tensor(prior_validation, dtype=torch.float32, device=device)
    validation_context = torch.as_tensor(
        validation.context if validation.context is not None else np.zeros((len(validation.x), model.context_features), dtype=np.float32),
        dtype=torch.float32, device=device,
    )
    validation_y = np.asarray(validation.y, dtype=np.float32)
    anchor = None
    if encoder_anchor is not None:
        anchor = {key: value.detach().to(device) for key, value in encoder_anchor.items()}
    best_score = float("inf")
    best_epoch = int(epochs)
    best_state = None
    best_unfrozen_score = float("inf")
    best_unfrozen_epoch = None
    best_unfrozen_state = None
    stale = 0
    epoch = 0
    for epoch in range(1, int(epochs) + 1):
        if warmup and epoch == warmup + 1:
            set_encoder_trainable(True)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=2.5e-4, weight_decay=1.0e-4)
        model.train()
        order = rng.permutation(len(train.x))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            output = model.forward_target(train_x[index], train_prior[index], train_context[index]).rul
            loss = F.smooth_l1_loss(output, train_y[index], beta=0.05)
            if anchor is not None and encoder_anchor_weight > 0.0 and not frozen_encoder:
                anchor_penalty = torch.zeros((), dtype=loss.dtype, device=device)
                for name, parameter in model.encoder.named_parameters():
                    anchor_penalty = anchor_penalty + F.mse_loss(parameter, anchor[name])
                loss = loss + float(encoder_anchor_weight) * anchor_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model.forward_target(validation_x, validation_prior, validation_context).rul.detach().cpu().numpy()
        score = rmse(validation_y, prediction)
        if score < best_score - 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = _best_state(model)
            stale = 0
        else:
            stale += 1
        if not frozen_encoder and epoch > warmup and score < best_unfrozen_score - 1.0e-6:
            best_unfrozen_score = score
            best_unfrozen_epoch = epoch
            best_unfrozen_state = _best_state(model)
        if stale >= patience:
            break
    selected_state = best_unfrozen_state if require_unfrozen_checkpoint else best_state
    selected_epoch = best_unfrozen_epoch if require_unfrozen_checkpoint else best_epoch
    selected_score = best_unfrozen_score if require_unfrozen_checkpoint else best_score
    if selected_state is None:
        raise RuntimeError("No eligible validation checkpoint was retained")
    model.load_state_dict(selected_state)
    model.eval()
    return model, {
        "best_validation_normalized_rmse": float(selected_score),
        "best_epoch": int(selected_epoch),
        "epochs_ran": int(epoch),
        "frozen_encoder": bool(frozen_encoder),
        "requested_warmup_frozen_epochs": int(warmup_frozen_epochs),
        "warmup_frozen_epochs": int(warmup),
        "encoder_unfreeze_epoch": (None if frozen_encoder else int(warmup + 1)),
        "encoder_anchor_weight": float(encoder_anchor_weight),
        "encoder_was_unfrozen": bool(not frozen_encoder and warmup < epoch),
        "require_unfrozen_checkpoint": bool(require_unfrozen_checkpoint),
        "best_checkpoint_after_encoder_unfreeze": bool(
            not frozen_encoder and best_unfrozen_epoch is not None and selected_epoch > warmup
        ),
        "residual_target": True,
        "bounded_residual_range": [-0.5, 0.5],
    }

@torch.no_grad()
def _predict_residual_model(
    model: TrendResidualRULModel,
    data: PreparedData,
    prior: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    p = torch.as_tensor(prior, dtype=torch.float32, device=device)
    context = torch.as_tensor(
        data.context if data.context is not None else np.zeros((len(data.x), model.context_features), dtype=np.float32),
        dtype=torch.float32, device=device
    )
    prediction = model.forward_target(x, p, context).rul.detach().cpu().numpy()
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Trend residual model produced non-finite output")
    return (np.maximum(prediction, 0.0) * data.scale).astype(np.float32)


def _select_residual_blend_weight(
    validation: PreparedData,
    prior_validation: np.ndarray,
    model_validation: np.ndarray,
    *,
    max_weight: float = 1.0,
) -> tuple[float, float]:
    """Select a causal prior/model blend using chronological validation only."""
    prior_raw = np.asarray(prior_validation, dtype=np.float32) * validation.scale
    model_raw = np.asarray(model_validation, dtype=np.float32)
    return _select_residual_blend_weight_raw(
        validation,
        prior_raw,
        model_raw,
        max_weight=max_weight,
    )


def _select_residual_blend_weight_raw(
    validation: PreparedData,
    prior_raw: np.ndarray,
    model_raw: np.ndarray,
    *,
    max_weight: float = 1.0,
) -> tuple[float, float]:
    """Select a prior/model blend from raw-unit validation predictions.

    The caller may pass a seed ensemble here.  Keeping this helper in raw
    units makes it explicit that the selection is performed once per outer
    fold, after seed aggregation, and never on the outer test trajectory.
    """
    prior_raw = np.asarray(prior_raw, dtype=np.float32)
    model_raw = np.asarray(model_raw, dtype=np.float32)
    if prior_raw.shape != validation.y.shape or model_raw.shape != validation.y.shape:
        raise ValueError("Residual blend predictions must match validation labels")
    if not 0.0 <= float(max_weight) <= 1.0:
        raise ValueError("Residual blend max_weight must be in [0, 1]")
    truth = validation.y * validation.scale
    best_weight, best_score = 0.0, rmse(truth, prior_raw)
    for weight in np.linspace(0.0, float(max_weight), 21):
        prediction = prior_raw + float(weight) * (model_raw - prior_raw)
        score = rmse(truth, prediction)
        if score < best_score - 1.0e-7:
            best_weight, best_score = float(weight), float(score)
    return best_weight, float(best_score)


def _mix_transfer_with_trend(
    transfer_prediction: np.ndarray,
    trend_prediction: np.ndarray,
    *,
    transfer_weight: float,
) -> np.ndarray:
    """Make a predeclared transfer/trend hybrid without reading outer labels."""
    if not 0.0 <= float(transfer_weight) <= 1.0:
        raise ValueError("Calibrated transfer weight must be in [0, 1]")
    transfer_prediction = np.asarray(transfer_prediction, dtype=np.float32)
    trend_prediction = np.asarray(trend_prediction, dtype=np.float32)
    if transfer_prediction.shape != trend_prediction.shape:
        raise ValueError("Transfer and trend predictions must have identical shapes")
    output = (
        float(transfer_weight) * transfer_prediction
        + (1.0 - float(transfer_weight)) * trend_prediction
    ).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Calibrated transfer produced non-finite output")
    return output


def _fit_validation_affine_calibrator(
    validation: PreparedData,
    prediction_raw: np.ndarray,
    *,
    slope_bounds: tuple[float, float] = AFFINE_SLOPE_BOUNDS,
    intercept_fraction: float = AFFINE_INTERCEPT_FRACTION,
) -> dict[str, object]:
    """Fit a bounded raw-RUL calibration from chronological validation only.

    This is a target-side output adapter, not a target-only predictor. The
    identity mapping remains a candidate, so the adapter cannot make the
    validation score worse by construction. Bounds are fixed before outer
    evaluation and prevent a one-trajectory calibration from replacing the
    transfer model.
    """
    prediction_raw = np.asarray(prediction_raw, dtype=np.float64)
    truth = np.asarray(validation.y, dtype=np.float64) * float(validation.scale)
    if prediction_raw.shape != validation.y.shape:
        raise ValueError("Affine calibration predictions must match validation labels")
    if not np.all(np.isfinite(prediction_raw)) or not np.all(np.isfinite(truth)):
        raise FloatingPointError("Affine calibration requires finite validation values")
    lower, upper = (float(slope_bounds[0]), float(slope_bounds[1]))
    if not 0.0 < lower <= upper:
        raise ValueError("Affine calibration slope bounds must satisfy 0 < lower <= upper")
    if float(intercept_fraction) < 0.0:
        raise ValueError("Affine calibration intercept fraction must be non-negative")

    centered = prediction_raw - float(np.mean(prediction_raw))
    variance = float(np.dot(centered, centered))
    if variance <= 1.0e-12:
        fitted_slope = 1.0
    else:
        fitted_slope = float(np.dot(centered, truth - float(np.mean(truth))) / variance)
    fitted_slope = float(np.clip(fitted_slope, lower, upper))
    fitted_intercept = float(np.mean(truth) - fitted_slope * np.mean(prediction_raw))
    intercept_limit = float(intercept_fraction) * max(float(validation.scale), 1.0)
    fitted_intercept = float(np.clip(fitted_intercept, -intercept_limit, intercept_limit))

    affine_prediction = fitted_slope * prediction_raw + fitted_intercept
    identity_score = rmse(truth, prediction_raw)
    affine_score = rmse(truth, affine_prediction)
    minimum_improvement_raw = max(1.0e-5, 1.0e-4 * max(identity_score, 1.0))
    use_affine = bool(affine_score < identity_score - minimum_improvement_raw)
    selected_slope = fitted_slope if use_affine else 1.0
    selected_intercept = fitted_intercept if use_affine else 0.0
    selected_score = affine_score if use_affine else identity_score
    return {
        "selected_slope": float(selected_slope),
        "selected_intercept_raw": float(selected_intercept),
        "fitted_slope": float(fitted_slope),
        "fitted_intercept_raw": float(fitted_intercept),
        "slope_bounds": [lower, upper],
        "intercept_limit_raw": float(intercept_limit),
        "identity_validation_raw_rmse": float(identity_score),
        "affine_validation_raw_rmse": float(affine_score),
        "minimum_required_improvement_raw": float(minimum_improvement_raw),
        "selected_validation_raw_rmse": float(selected_score),
        "selected_mapping": "affine" if use_affine else "identity",
        "selection": "chronological_validation_only",
        "holdout_used": False,
    }


def _apply_validation_affine_calibrator(
    prediction_raw: np.ndarray,
    calibrator: dict[str, object],
) -> np.ndarray:
    prediction_raw = np.asarray(prediction_raw, dtype=np.float32)
    output = (
        float(calibrator["selected_slope"]) * prediction_raw
        + float(calibrator["selected_intercept_raw"])
    ).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Affine-calibrated transfer produced non-finite output")
    return output


def _select_transfer_trend_weight(
    validation: PreparedData,
    transfer_prediction: np.ndarray,
    trend_prediction: np.ndarray,
    *,
    min_weight: float = CALIBRATED_MIN_TRANSFER_WEIGHT,
    max_weight: float = 1.0,
) -> tuple[float, float]:
    """Select a transfer/Huber-trend mixture using validation labels only."""
    if not 0.0 <= float(min_weight) <= float(max_weight) <= 1.0:
        raise ValueError("Calibrated transfer weight bounds must satisfy 0 <= min <= max <= 1")
    transfer_prediction = np.asarray(transfer_prediction, dtype=np.float32)
    trend_prediction = np.asarray(trend_prediction, dtype=np.float32)
    if transfer_prediction.shape != validation.y.shape or trend_prediction.shape != validation.y.shape:
        raise ValueError("Calibrated validation predictions must match validation labels")
    truth = validation.y * validation.scale
    weights = np.linspace(float(min_weight), float(max_weight), 4)
    best_weight = float(weights[0])
    best_score = float("inf")
    for weight in weights:
        prediction = weight * transfer_prediction + (1.0 - weight) * trend_prediction
        score = rmse(truth, prediction)
        # Prefer more source transfer on a validation tie.
        if score < best_score - 1.0e-7 or (
            abs(score - best_score) <= 1.0e-7 and float(weight) > best_weight
        ):
            best_weight, best_score = float(weight), float(score)
    return best_weight, float(best_score)


def _select_transfer_ridge_weight(
    validation: PreparedData,
    transfer_prediction: np.ndarray,
    ridge_prediction: np.ndarray,
    *,
    min_weight: float = CALIBRATED_MIN_TRANSFER_WEIGHT,
    max_weight: float = 1.0,
) -> tuple[float, float]:
    """Select a transfer/Ridge mixture from chronological validation only.

    The Ridge component is target-only and is therefore reported as part of a
    separate hybrid diagnostic. Keeping a non-zero transfer floor prevents
    this method from silently collapsing into the target-only baseline.
    """
    if not 0.0 <= float(min_weight) <= float(max_weight) <= 1.0:
        raise ValueError("Transfer/Ridge weight bounds must satisfy 0 <= min <= max <= 1")
    transfer_prediction = np.asarray(transfer_prediction, dtype=np.float32)
    ridge_prediction = np.asarray(ridge_prediction, dtype=np.float32)
    if transfer_prediction.shape != validation.y.shape or ridge_prediction.shape != validation.y.shape:
        raise ValueError("Transfer/Ridge validation predictions must match labels")
    if not np.all(np.isfinite(transfer_prediction)) or not np.all(np.isfinite(ridge_prediction)):
        raise FloatingPointError("Transfer/Ridge validation predictions must be finite")
    truth = validation.y * validation.scale
    weights = np.linspace(float(min_weight), float(max_weight), 4)
    best_weight = float(weights[0])
    best_score = float("inf")
    for weight in weights:
        prediction = weight * transfer_prediction + (1.0 - weight) * ridge_prediction
        score = rmse(truth, prediction)
        # Prefer more source transfer on a validation tie.
        if score < best_score - 1.0e-7 or (
            abs(score - best_score) <= 1.0e-7 and float(weight) > best_weight
        ):
            best_weight, best_score = float(weight), float(score)
    return best_weight, float(best_score)


def _prepare_inner_group_loo_target(
    table: SimulationTable,
    *,
    fit_groups: Sequence[str],
    heldout_group: str,
) -> tuple[PreparedData, PreparedData, dict[str, object]]:
    """Prepare one inner group-LOO fold without materializing an outer test set."""
    train_groups = tuple(str(group) for group in fit_groups)
    heldout = str(heldout_group)
    if len(train_groups) < 1 or heldout in set(train_groups):
        raise ValueError("Inner group-LOO train and held-out groups must be disjoint")
    mean, std, label_scale = _target_stats(table, train_groups)
    train = _make_target_windows(
        table, train_groups, mean=mean, std=std, label_scale=label_scale
    )
    heldout_data = _make_target_windows(
        table, (heldout,), mean=mean, std=std, label_scale=label_scale
    )
    return train, heldout_data, {
        "fit_groups": list(train_groups),
        "inner_heldout_group": heldout,
        "target_scaler_fit_groups": list(train_groups),
        "target_label_scale_fit_groups": list(train_groups),
        "heldout_in_scaler_fit": False,
        "outer_holdout_materialized": False,
    }


def _select_group_balanced_mixture_weight(
    truth_raw: np.ndarray,
    primary_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_weight: float,
    max_weight: float,
    grid_size: int,
) -> tuple[float, float, dict[str, object]]:
    """Select a mixture by mean group RMSE, avoiding long-group dominance.

    All arrays are out-of-fold predictions from outer-fit groups only. Ties use
    the smaller primary-model weight, a predeclared conservative rule.
    """
    truth = np.asarray(truth_raw, dtype=np.float64)
    primary = np.asarray(primary_prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=str)
    if (
        truth.ndim != 1
        or primary.shape != truth.shape
        or baseline.shape != truth.shape
        or groups.shape != truth.shape
        or len(truth) == 0
    ):
        raise ValueError("Inner group-LOO mixture arrays must be non-empty matching vectors")
    if not 0.0 <= float(min_weight) <= float(max_weight) <= 1.0:
        raise ValueError("Inner group-LOO mixture bounds must satisfy 0 <= min <= max <= 1")
    if int(grid_size) < 2:
        raise ValueError("Inner group-LOO mixture grid must contain at least two values")
    if not all(np.all(np.isfinite(values)) for values in (truth, primary, baseline)):
        raise FloatingPointError("Inner group-LOO mixture selection requires finite arrays")
    names = tuple(sorted(set(groups.tolist()), key=_natural_group))
    if len(names) < 2:
        raise ValueError("Inner group-LOO mixture selection requires at least two held-out groups")
    weights = np.linspace(float(min_weight), float(max_weight), int(grid_size))
    candidate_scores: list[dict[str, object]] = []
    best_weight = float(weights[0])
    best_score = float("inf")
    for weight in weights:
        prediction = float(weight) * primary + (1.0 - float(weight)) * baseline
        per_group = {
            name: float(rmse(truth[groups == name], prediction[groups == name]))
            for name in names
        }
        macro_score = float(np.mean(list(per_group.values())))
        candidate_scores.append({
            "primary_weight": float(weight),
            "macro_raw_rmse": macro_score,
            "per_group_raw_rmse": per_group,
        })
        if macro_score < best_score - 1.0e-7:
            best_weight, best_score = float(weight), macro_score
    selected = next(item for item in candidate_scores if item["primary_weight"] == best_weight)
    return best_weight, best_score, {
        "selection": "inner_leave_one_fit_group_out",
        "selection_objective": "mean_inner_group_raw_rmse",
        "tie_break": "smaller_primary_weight",
        "inner_groups": list(names),
        "inner_group_window_counts": {
            name: int(np.sum(groups == name)) for name in names
        },
        "weight_grid": [float(value) for value in weights],
        "candidates": candidate_scores,
        "selected_per_group_raw_rmse": selected["per_group_raw_rmse"],
        "outer_holdout_used": False,
        "outer_validation_used_for_selection": False,
    }


def _select_inner_group_loo_weights(
    *,
    source_bundle,
    source_models: dict[int, CrossDomainRULModel],
    table: SimulationTable,
    outer_holdout: str,
    n_shots: int,
    fit_groups: Sequence[str],
    seeds: Sequence[int],
    epochs: int,
    device: str,
    prior_policy: str,
    use_context: bool,
    max_blend_weight: float,
    transfer_residual_head: bool,
    encoder_anchor_weight: float,
    transfer_warmup_epochs: int,
    target_include_deltas: bool,
    target_adapter_alignment_steps: int,
) -> dict[str, object]:
    """Choose residual and baseline mixtures from outer-fit group OOF predictions."""
    groups = tuple(str(group) for group in fit_groups)
    if len(groups) < 2:
        return {
            "available": False,
            "selection": "chronological_validation_fallback",
            "reason": "inner_group_loo_requires_at_least_two_outer_fit_groups",
            "outer_fit_groups": list(groups),
            "outer_holdout": str(outer_holdout),
            "outer_holdout_used": False,
            "outer_validation_used_for_selection": True,
        }

    entries: list[dict[str, object]] = []
    inner_folds: list[dict[str, object]] = []
    for inner_index, inner_heldout in enumerate(groups):
        inner_fit_groups = tuple(group for group in groups if group != inner_heldout)
        inner_train, inner_validation, inner_audit = _prepare_inner_group_loo_target(
            table, fit_groups=inner_fit_groups, heldout_group=inner_heldout
        )
        if target_include_deltas:
            inner_train = _append_target_deltas(inner_train)
            inner_validation = _append_target_deltas(inner_validation)
        priors, prior_diag = _select_prior(
            inner_train, inner_validation, inner_validation, policy=prior_policy
        )
        predictions: dict[str, list[np.ndarray]] = {method: [] for method in SEED_METHODS}
        fit_diagnostics: dict[str, list[dict[str, object]]] = {method: [] for method in SEED_METHODS}
        for method in SEED_METHODS:
            for seed in seeds:
                training_seed = (
                    int(seed)
                    + int(n_shots) * 1000
                    + _fold_number(str(outer_holdout)) * 10000
                    + (inner_index + 1) * 100000
                )
                model = _new_model(
                    int(inner_train.x.shape[2]),
                    training_seed,
                    context_features=(len(table.feature_names) if use_context else 0),
                )
                source_model: CrossDomainRULModel | None = None
                if method in {"trend_residual_transfer", "trend_residual_frozen"}:
                    source_model = source_models.get(int(seed))
                    if source_model is None:
                        raise ValueError(f"Missing source model for seed {seed}")
                    copy_shared_encoder_to_trend_residual(
                        source_model, model, copy_residual_head=transfer_residual_head
                    )
                    adapter_alignment = _align_target_adapter_to_source(
                        source_model,
                        model,
                        source_bundle.train,
                        inner_train,
                        device=device,
                        seed=training_seed,
                        steps=target_adapter_alignment_steps,
                    )
                else:
                    adapter_alignment = {
                        "enabled": False,
                        "reason": "scratch_target_only_control",
                        "target_labels_used": False,
                        "validation_or_holdout_used": False,
                    }
                model, diagnostic = _train_residual_model(
                    model,
                    inner_train,
                    inner_validation,
                    priors["train"],
                    priors["validation"],
                    device=device,
                    seed=training_seed,
                    epochs=epochs,
                    frozen_encoder=method == "trend_residual_frozen",
                    warmup_frozen_epochs=(
                        int(transfer_warmup_epochs)
                        if method == "trend_residual_transfer" else 0
                    ),
                    require_unfrozen_checkpoint=method == "trend_residual_transfer",
                    encoder_anchor=(
                        source_model.encoder.state_dict()
                        if method == "trend_residual_transfer" and source_model is not None else None
                    ),
                    encoder_anchor_weight=(
                        float(encoder_anchor_weight)
                        if method == "trend_residual_transfer" else 0.0
                    ),
                )
                predictions[method].append(_predict_residual_model(
                    model, inner_validation, priors["validation"], device=device
                ))
                fit_diagnostics[method].append({
                    "seed": int(seed),
                    "training_seed": int(training_seed),
                    "best_epoch": int(diagnostic["best_epoch"]),
                    "target_adapter_alignment": adapter_alignment,
                })
        ridge_alpha, ridge_validation_rmse = _select_ridge_alpha(inner_train, inner_validation)
        ridge_prediction = _predict_ridge(inner_validation, _fit_ridge(inner_train, ridge_alpha))
        endpoint_scale = max(float(np.max(inner_train.endpoints)), 1.0)
        huber_prediction = _predict_trend(
            _fit_huber_trend(inner_train, endpoint_scale, 1.35),
            inner_validation,
            degree=1,
            endpoint_scale=endpoint_scale,
        )
        entry = {
            "group": str(inner_heldout),
            "truth": (inner_validation.y * inner_validation.scale).astype(np.float32),
            "prior": (priors["validation"] * inner_validation.scale).astype(np.float32),
            "huber": huber_prediction.astype(np.float32),
            "ridge": ridge_prediction.astype(np.float32),
            **{
                method: np.mean(np.stack(values), axis=0).astype(np.float32)
                for method, values in predictions.items()
            },
        }
        entries.append(entry)
        inner_folds.append({
            **inner_audit,
            "inner_heldout_labels_used_for_early_stopping": True,
            "prior": prior_diag,
            "ridge_alpha": float(ridge_alpha),
            "ridge_inner_validation_raw_rmse": float(ridge_validation_rmse),
            "seed_fit_diagnostics": fit_diagnostics,
        })

    group_ids = np.concatenate([
        np.repeat(str(entry["group"]), len(np.asarray(entry["truth"])))
        for entry in entries
    ])
    truth = np.concatenate([np.asarray(entry["truth"]) for entry in entries]).astype(np.float32)

    def values(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(entry[name]) for entry in entries]).astype(np.float32)

    residual_blend: dict[str, dict[str, object]] = {}
    corrected: dict[str, np.ndarray] = {}
    for method in SEED_METHODS:
        weight, score, selection_audit = _select_group_balanced_mixture_weight(
            truth, values(method), values("prior"), group_ids,
            min_weight=0.0, max_weight=max_blend_weight, grid_size=21,
        )
        corrected[method] = _mix_transfer_with_trend(
            values(method), values("prior"), transfer_weight=weight
        )
        residual_blend[method] = {
            "selected_weight": float(weight),
            "inner_macro_raw_rmse": float(score),
            **selection_audit,
        }

    def hybrid(baseline: str) -> dict[str, dict[str, object]]:
        selected: dict[str, dict[str, object]] = {}
        for method in ("trend_residual_transfer", "trend_residual_scratch"):
            weight, score, selection_audit = _select_group_balanced_mixture_weight(
                truth, corrected[method], values(baseline), group_ids,
                min_weight=CALIBRATED_MIN_TRANSFER_WEIGHT, max_weight=1.0, grid_size=4,
            )
            selected[method] = {
                "selected_weight": float(weight),
                "inner_macro_raw_rmse": float(score),
                **selection_audit,
            }
        return selected

    return {
        "available": True,
        "selection": "inner_leave_one_fit_group_out",
        "outer_fit_groups": list(groups),
        "outer_holdout": str(outer_holdout),
        "outer_holdout_used": False,
        "outer_validation_used_for_selection": False,
        "inner_folds": inner_folds,
        "residual_blend": residual_blend,
        "trend_hybrid": hybrid("huber"),
        "ridge_hybrid": hybrid("ridge"),
    }


def _ensemble_method_order() -> tuple[str, ...]:
    return (
        "trend_residual_transfer_ensemble",
        f"{CALIBRATED_TRANSFER_METHOD}_ensemble",
        f"{AFFINE_CALIBRATED_TRANSFER_METHOD}_ensemble",
        f"{RIDGE_CALIBRATED_TRANSFER_METHOD}_ensemble",
        "trend_residual_frozen_ensemble",
        "trend_residual_scratch_ensemble",
        f"{CALIBRATED_SCRATCH_METHOD}_ensemble",
        f"{AFFINE_CALIBRATED_SCRATCH_METHOD}_ensemble",
        f"{RIDGE_CALIBRATED_SCRATCH_METHOD}_ensemble",
        "ridge_ensemble",
        "trend_linear_ensemble",
        "trend_quadratic_ensemble",
        "trend_huber_ensemble",
    )


def _summarize_selected(
    source: str,
    n_shots: int | str,
    selected: Sequence[dict[str, object]],
    *,
    aggregation_scope: str | None = None,
) -> list[dict[str, object]]:
    """Summarize one predeclared outer-fold evaluation scope fairly."""
    rows_by_method: dict[str, list[dict[str, object]]] = {}
    for row in selected:
        rows_by_method.setdefault(str(row["method"]), []).append(row)

    def fold_key(row: dict[str, object]) -> tuple[str, str]:
        return str(row["n_shots"]), str(row["outer_holdout"])

    ridge_by_holdout = {
        fold_key(row): row
        for row in rows_by_method.get("ridge_ensemble", [])
    }
    trend_by_holdout: dict[tuple[str, str], list[float]] = {}
    for method in ("trend_linear_ensemble", "trend_quadratic_ensemble", "trend_huber_ensemble"):
        for row in rows_by_method.get(method, []):
            trend_by_holdout.setdefault(fold_key(row), []).append(float(row["raw_rmse"]))

    summaries: list[dict[str, object]] = []
    for method in _ensemble_method_order():
        method_rows = rows_by_method.get(method, [])
        if not method_rows:
            continue
        fold_values = np.asarray([float(row["raw_rmse"]) for row in method_rows], dtype=np.float64)
        reference_method = MATCHED_SCRATCH_REFERENCE.get(method, "trend_residual_scratch_ensemble")
        reference_by_holdout = {
            fold_key(row): row
            for row in rows_by_method.get(reference_method, [])
        }
        matched_reference_available = all(
            fold_key(row) in reference_by_holdout for row in method_rows
        )
        macro = float(np.mean(fold_values))
        if matched_reference_available:
            scratch_values = np.asarray(
                [float(reference_by_holdout[fold_key(row)]["raw_rmse"]) for row in method_rows],
                dtype=np.float64,
            )
            scratch_macro: float | None = float(np.mean(scratch_values))
            improvement: float | None = float(100.0 * (scratch_macro - macro) / max(scratch_macro, 1.0e-12))
            wins: int | None = int(np.sum(fold_values < scratch_values))
            losses_or_ties: int | None = int(len(method_rows) - wins)
            pass_matched_scratch = bool(macro <= 0.95 * scratch_macro and wins > len(method_rows) / 2.0)
        else:
            scratch_macro = None
            improvement = None
            wins = None
            losses_or_ties = None
            pass_matched_scratch = False

        ridge_available = all(fold_key(row) in ridge_by_holdout for row in method_rows)
        ridge_values = np.asarray(
            [float(ridge_by_holdout[fold_key(row)]["raw_rmse"]) for row in method_rows],
            dtype=np.float64,
        ) if ridge_available else np.asarray([], dtype=np.float64)
        ridge_macro: float | None = float(np.mean(ridge_values)) if ridge_available else None
        macro_vs_ridge = float(100.0 * (ridge_macro - macro) / max(ridge_macro, 1.0e-12)) if ridge_macro is not None else None
        best_trend_available = all(fold_key(row) in trend_by_holdout for row in method_rows)
        best_trend_values = np.asarray(
            [min(trend_by_holdout[fold_key(row)]) for row in method_rows],
            dtype=np.float64,
        ) if best_trend_available else np.asarray([], dtype=np.float64)
        best_trend_macro: float | None = float(np.mean(best_trend_values)) if best_trend_available else None
        macro_vs_best_trend = float(100.0 * (best_trend_macro - macro) / max(best_trend_macro, 1.0e-12)) if best_trend_macro is not None else None
        trend_wins: int | None = int(np.sum(fold_values < best_trend_values)) if best_trend_available else None
        beats_ridge = bool(ridge_macro is not None and macro < ridge_macro)
        beats_best_trend = bool(best_trend_macro is not None and macro < best_trend_macro)
        is_transfer = method in TRANSFER_EVIDENCE_METHODS
        strict_pass = bool(is_transfer and matched_reference_available and pass_matched_scratch and beats_ridge and beats_best_trend)
        if not is_transfer:
            acceptance_reason = "benchmark or ablation comparator"
        elif not matched_reference_available:
            acceptance_reason = f"diagnostic: missing matched scratch comparator {reference_method}; no transfer acceptance is inferred"
        elif strict_pass:
            acceptance_reason = "strict positive evidence: >=5% macro improvement over matched scratch, majority matched-fold wins, and lower macro RMSE than Ridge and best trend"
        else:
            acceptance_reason = "diagnostic: did not jointly meet matched-scratch improvement, majority-fold, Ridge, and deterministic-trend criteria"
        item: dict[str, object] = {
            "source": source,
            "n_shots": n_shots,
            "method": method,
            "outer_folds": len(method_rows),
            "macro_raw_rmse": macro,
            "macro_mae": float(np.mean([float(row["mae"]) for row in method_rows])),
            "macro_bias": float(np.mean([float(row["bias"]) for row in method_rows])),
            "macro_normalized_rmse": float(np.mean([float(row["normalized_rmse"]) for row in method_rows])),
            "matched_scratch_method": reference_method,
            "matched_scratch_comparator_available": matched_reference_available,
            "matched_scratch_macro_raw_rmse": scratch_macro,
            "improvement_vs_matched_scratch_pct": improvement,
            "matched_outer_fold_wins": wins,
            "matched_outer_fold_losses_or_ties": losses_or_ties,
            "scratch_macro_raw_rmse": scratch_macro,
            "improvement_vs_scratch_pct": improvement,
            "outer_fold_wins_vs_scratch": wins,
            "outer_fold_losses_or_ties_vs_scratch": losses_or_ties,
            "ridge_macro_raw_rmse": ridge_macro,
            "macro_vs_ridge_pct": macro_vs_ridge,
            "best_deterministic_trend_macro_raw_rmse": best_trend_macro,
            "macro_vs_best_trend_pct": macro_vs_best_trend,
            "trend_fold_win_count": trend_wins,
            "beats_ridge": beats_ridge,
            "beats_best_trend": beats_best_trend,
            "matched_scratch_criterion_pass": pass_matched_scratch,
            "strict_positive_transfer_evidence": strict_pass,
            "positive_transfer_evidence": bool(method == "trend_residual_transfer_ensemble" and strict_pass),
            "hybrid_transfer_evidence": bool(method == f"{CALIBRATED_TRANSFER_METHOD}_ensemble" and strict_pass),
            "affine_calibrated_transfer_evidence": bool(method == f"{AFFINE_CALIBRATED_TRANSFER_METHOD}_ensemble" and strict_pass),
            "ridge_calibrated_transfer_evidence": bool(method == f"{RIDGE_CALIBRATED_TRANSFER_METHOD}_ensemble" and strict_pass),
            "baseline_comparison_pass": strict_pass,
            "acceptance_reason": acceptance_reason,
        }
        if aggregation_scope is not None:
            item["aggregation_scope"] = aggregation_scope
        summaries.append(item)
    return summaries


def _summarize(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for source in sorted({str(row["source"]) for row in rows}):
        for n_shots in TARGET_N_SHOTS:
            selected = [
                row for row in rows
                if row["source"] == source
                and int(row["n_shots"]) == n_shots
                and str(row["method"]).endswith("_ensemble")
            ]
            summaries.extend(_summarize_selected(source, n_shots, selected))
    return summaries


def _summarize_all_shots(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate all predeclared N=1/2/3 outer-fold units without selection."""
    summaries: list[dict[str, object]] = []
    for source in sorted({str(row["source"]) for row in rows}):
        selected = [
            row for row in rows
            if row["source"] == source
            and int(row["n_shots"]) in TARGET_N_SHOTS
            and str(row["method"]).endswith("_ensemble")
        ]
        summaries.extend(_summarize_selected(
            source,
            "all_predeclared",
            selected,
            aggregation_scope="N=1,2,3 x all requested outer holdouts",
        ))
    return summaries


def _run_outer(
    source_name: str,
    source_bundle,
    source_models: dict[int, torch.nn.Module],
    table: SimulationTable,
    *,
    seeds: Sequence[int],
    epochs: int,
    device: str,
    holdouts: Sequence[str],
    prior_policy: str,
    use_context: bool,
    max_blend_weight: float,
    blend_selection_policy: str,
    transfer_residual_head: bool,
    calibrated_transfer_weight: float,
    calibrated_transfer_weight_policy: str,
    encoder_anchor_weight: float,
    transfer_warmup_epochs: int,
    target_include_deltas: bool,
    target_adapter_alignment_steps: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if blend_selection_policy not in BLEND_SELECTION_POLICIES:
        raise ValueError(
            "Blend selection policy must be one of "
            f"{BLEND_SELECTION_POLICIES}"
        )
    if calibrated_transfer_weight_policy not in CALIBRATED_WEIGHT_POLICIES:
        raise ValueError(
            "Calibrated transfer weight policy must be one of "
            f"{CALIBRATED_WEIGHT_POLICIES}"
        )
    if calibrated_transfer_weight < CALIBRATED_MIN_TRANSFER_WEIGHT:
        raise ValueError(
            "Fixed calibrated transfer weight must preserve at least "
            f"{CALIBRATED_MIN_TRANSFER_WEIGHT:.2f} source transfer"
        )
    if transfer_warmup_epochs < 1:
        raise ValueError("Transfer warmup must be at least one epoch")
    if target_adapter_alignment_steps < 0:
        raise ValueError("Target adapter alignment steps must be non-negative")
    groups = tuple(sorted(set(table.group_ids.tolist()), key=_natural_group))
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for holdout in holdouts:
        for n_shots in TARGET_N_SHOTS:
            split = outer_schedule(groups, n_shots)[holdout]
            train, validation, test, audit = prepare_outer_target(
                table,
                fit_groups=split["fit_groups"],
                validation_groups=split["validation_groups"],
                test_groups=split["test_groups"],
            )
            if target_include_deltas:
                train = _append_target_deltas(train)
                validation = _append_target_deltas(validation)
                test = _append_target_deltas(test)
            priors, prior_diag = _select_prior(
                train,
                validation,
                test,
                policy=prior_policy,
            )
            prior_train = priors["train"]
            prior_validation = priors["validation"]
            prior_test = priors["test"]
            inner_group_loo = (
                _select_inner_group_loo_weights(
                    source_bundle=source_bundle,
                    source_models=source_models,
                    table=table,
                    outer_holdout=holdout,
                    n_shots=n_shots,
                    fit_groups=split["fit_groups"],
                    seeds=seeds,
                    epochs=epochs,
                    device=device,
                    prior_policy=prior_policy,
                    use_context=use_context,
                    max_blend_weight=max_blend_weight,
                    transfer_residual_head=transfer_residual_head,
                    encoder_anchor_weight=encoder_anchor_weight,
                    transfer_warmup_epochs=transfer_warmup_epochs,
                    target_include_deltas=target_include_deltas,
                    target_adapter_alignment_steps=target_adapter_alignment_steps,
                )
                if blend_selection_policy == "inner_group_loo" else {
                    "available": False,
                    "selection": "chronological_validation_requested",
                    "reason": "explicit_validation_ablation",
                    "outer_fit_groups": list(split["fit_groups"]),
                    "outer_holdout": str(holdout),
                    "outer_holdout_used": False,
                    "outer_validation_used_for_selection": True,
                }
            )
            audit.update({
                "outer_holdout": holdout,
                "n_shots": n_shots,
                "trend_prior": prior_diag,
                "holdout_in_prior_fit": False,
                "full_life_min_max_used_for_target_input": False,
                "target_input_feature_transform": (
                    "raw_levels_plus_causal_first_differences"
                    if target_include_deltas else "raw_levels"
                ),
                "blend_selection_policy": blend_selection_policy,
                "inner_group_loo_selection": inner_group_loo,
            })
            audits[f"{holdout}:{n_shots}"] = audit
            method_predictions: dict[str, list[np.ndarray]] = {
                "trend_residual_transfer": [],
                CALIBRATED_TRANSFER_METHOD: [],
                AFFINE_CALIBRATED_TRANSFER_METHOD: [],
                RIDGE_CALIBRATED_TRANSFER_METHOD: [],
                CALIBRATED_SCRATCH_METHOD: [],
                AFFINE_CALIBRATED_SCRATCH_METHOD: [],
                RIDGE_CALIBRATED_SCRATCH_METHOD: [],
                "trend_residual_frozen": [],
                "trend_residual_scratch": [],
                "ridge": [],
                "trend_linear": [],
                "trend_quadratic": [],
                "trend_huber": [],
            }
            truth = test.y * test.scale
            seed_validation_predictions: dict[str, list[np.ndarray]] = {
                method: [] for method in SEED_METHODS
            }
            seed_test_predictions: dict[str, list[np.ndarray]] = {
                method: [] for method in SEED_METHODS
            }
            seed_fit_diagnostics: dict[str, list[dict[str, object]]] = {
                method: [] for method in SEED_METHODS
            }

            for method in SEED_METHODS:
                for seed in seeds:
                    training_seed = int(seed) + n_shots * 1000 + _fold_number(holdout) * 10000
                    model = _new_model(
                        int(train.x.shape[2]),
                        training_seed,
                        context_features=(len(table.feature_names) if use_context else 0),
                    )
                    if method in {"trend_residual_transfer", "trend_residual_frozen"}:
                        source_model = source_models.get(int(seed))
                        if source_model is None:
                            raise ValueError(f"Missing source model for seed {seed}")
                        copy_shared_encoder_to_trend_residual(
                            source_model, model, copy_residual_head=transfer_residual_head
                        )
                        adapter_alignment = _align_target_adapter_to_source(
                            source_model,
                            model,
                            source_bundle.train,
                            train,
                            device=device,
                            seed=training_seed,
                            steps=target_adapter_alignment_steps,
                        )
                    else:
                        adapter_alignment = {
                            "enabled": False,
                            "reason": "scratch_target_only_control",
                            "target_labels_used": False,
                            "validation_or_holdout_used": False,
                        }
                    model, fit_diag = _train_residual_model(
                        model,
                        train,
                        validation,
                        prior_train,
                        prior_validation,
                        device=device,
                        seed=training_seed,
                        epochs=epochs,
                        frozen_encoder=method == "trend_residual_frozen",
                        warmup_frozen_epochs=(
                            int(transfer_warmup_epochs)
                            if method == "trend_residual_transfer" else 0
                        ),
                        require_unfrozen_checkpoint=method == "trend_residual_transfer",
                        encoder_anchor=(
                            source_model.encoder.state_dict()
                            if method == "trend_residual_transfer" else None
                        ),
                        encoder_anchor_weight=(
                            float(encoder_anchor_weight)
                            if method == "trend_residual_transfer" else 0.0
                        ),
                    )
                    validation_prediction = _predict_residual_model(
                        model, validation, prior_validation, device=device
                    )
                    prediction = _predict_residual_model(model, test, prior_test, device=device)
                    seed_validation_predictions[method].append(validation_prediction)
                    seed_test_predictions[method].append(prediction)
                    seed_fit_diagnostics[method].append({
                        **fit_diag,
                        "training_seed": training_seed,
                        "trend_prior_method": prior_diag["selected_method"],
                        "raw_model_validation_available": True,
                        "target_adapter_alignment": adapter_alignment,
                    })

            # Aggregate seeds before selecting the residual correction. For N>=2,
            # weights come from inner group-LOO predictions restricted to outer
            # fit groups; N=1 transparently falls back to chronological validation.
            blend_audit: dict[str, object] = {}
            for method in SEED_METHODS:
                validation_ensemble = np.mean(
                    np.stack(seed_validation_predictions[method]), axis=0
                ).astype(np.float32)
                test_prior_raw = prior_test * test.scale
                validation_prior_raw = prior_validation * validation.scale
                if bool(inner_group_loo["available"]):
                    selected = inner_group_loo["residual_blend"][method]
                    blend_weight = float(selected["selected_weight"])
                    blend_validation_rmse = float(selected["inner_macro_raw_rmse"])
                    blend_audit[method] = {
                        "selection_unit": "seed_ensemble",
                        "selection_data": "fit_only_inner_group_loo",
                        "selected_weight": blend_weight,
                        "inner_macro_raw_rmse": blend_validation_rmse,
                        "max_weight": float(max_blend_weight),
                        "holdout_used": False,
                        "outer_validation_used_for_selection": False,
                        "inner_selection": selected,
                    }
                else:
                    blend_weight, blend_validation_rmse = _select_residual_blend_weight_raw(
                        validation,
                        validation_prior_raw,
                        validation_ensemble,
                        max_weight=max_blend_weight,
                    )
                    blend_audit[method] = {
                        "selection_unit": "seed_ensemble",
                        "selection_data": "chronological_validation_fallback",
                        "weight_grid": [float(value) for value in np.linspace(0.0, max_blend_weight, 21)],
                        "max_weight": float(max_blend_weight),
                        "selected_weight": float(blend_weight),
                        "validation_raw_rmse": float(blend_validation_rmse),
                        "holdout_used": False,
                        "outer_validation_used_for_selection": True,
                        "fallback_reason": str(inner_group_loo["reason"]),
                    }
                for seed, raw_prediction, fit_diag in zip(
                    seeds,
                    seed_test_predictions[method],
                    seed_fit_diagnostics[method],
                ):
                    prediction = (
                        test_prior_raw
                        + blend_weight * (raw_prediction - test_prior_raw)
                    ).astype(np.float32)
                    method_predictions[method].append(prediction)
                    rows.append({
                        "source": source_name,
                        "n_shots": n_shots,
                        "outer_holdout": holdout,
                        "method": method,
                        **metrics(truth, prediction, test.scale, method, seed),
                        "fit_groups": list(split["fit_groups"]),
                        "validation_groups": list(split["validation_groups"]),
                        "test_groups": list(split["test_groups"]),
                        "fit_diagnostics": {
                            **fit_diag,
                            "residual_blend_selection_unit": "seed_ensemble",
                            "residual_blend_weight": float(blend_weight),
                            "blend_selection_data": str(blend_audit[method]["selection_data"]),
                            "blend_selection_raw_rmse": float(blend_validation_rmse),
                        },
                    })
            audit["residual_blend_selection"] = blend_audit

            ridge_alpha, ridge_validation_rmse = _select_ridge_alpha(train, validation)
            ridge_beta = _fit_ridge(train, ridge_alpha)
            ridge_validation_prediction = _predict_ridge(validation, ridge_beta)
            ridge_prediction = _predict_ridge(test, ridge_beta)
            method_predictions["ridge"].append(ridge_prediction)
            rows.append({
                "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                "method": "ridge", **metrics(truth, ridge_prediction, test.scale, "ridge", None),
                "fit_groups": list(split["fit_groups"]), "validation_groups": list(split["validation_groups"]),
                "test_groups": list(split["test_groups"]),
                "fit_diagnostics": {"family": "target_only_ridge", "alpha": ridge_alpha, "validation_raw_rmse": ridge_validation_rmse},
            })

            linear_beta = _fit_trend(train, degree=1, endpoint_scale=max(float(np.max(train.endpoints)), 1.0))
            quadratic_beta = _fit_trend(train, degree=2, endpoint_scale=max(float(np.max(train.endpoints)), 1.0), alpha=1.0e-2)
            huber_beta = _fit_huber_trend(train, max(float(np.max(train.endpoints)), 1.0), 1.35)
            trend_betas = {"trend_linear": linear_beta, "trend_quadratic": quadratic_beta, "trend_huber": huber_beta}
            trend_degrees = {"trend_linear": 1, "trend_quadratic": 2, "trend_huber": 1}
            for method in TREND_METHODS:
                endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
                prediction = _predict_trend(trend_betas[method], test, trend_degrees[method], endpoint_scale)
                method_predictions[method].append(prediction)
                rows.append({
                    "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                    "method": method, **metrics(truth, prediction, test.scale, method, None),
                    "fit_groups": list(split["fit_groups"]), "validation_groups": list(split["validation_groups"]),
                    "test_groups": list(split["test_groups"]),
                    "fit_diagnostics": {"family": "causal_target_time_trend", "degree": trend_degrees[method], "trend_prior_selected_for_residual": prior_diag["selected_method"]},
                })

            test_prior_raw = prior_test * test.scale
            validation_prior_raw = prior_validation * validation.scale

            def _blended_validation_ensemble(method: str) -> np.ndarray:
                return np.mean(
                    np.stack([
                        validation_prior_raw
                        + float(blend_audit[method]["selected_weight"])
                        * (prediction - validation_prior_raw)
                        for prediction in seed_validation_predictions[method]
                    ]),
                    axis=0,
                ).astype(np.float32)

            transfer_ensemble = np.mean(
                np.stack(method_predictions["trend_residual_transfer"]), axis=0
            ).astype(np.float32)
            scratch_ensemble = np.mean(
                np.stack(method_predictions["trend_residual_scratch"]), axis=0
            ).astype(np.float32)
            transfer_validation_ensemble = _blended_validation_ensemble(
                "trend_residual_transfer"
            )
            scratch_validation_ensemble = _blended_validation_ensemble(
                "trend_residual_scratch"
            )

            affine_calibrator = _fit_validation_affine_calibrator(
                validation,
                transfer_validation_ensemble,
            )
            affine_calibrated_prediction = _apply_validation_affine_calibrator(
                transfer_ensemble,
                affine_calibrator,
            )
            method_predictions[AFFINE_CALIBRATED_TRANSFER_METHOD].append(
                affine_calibrated_prediction
            )
            scratch_affine_calibrator = _fit_validation_affine_calibrator(
                validation,
                scratch_validation_ensemble,
            )
            scratch_affine_prediction = _apply_validation_affine_calibrator(
                scratch_ensemble,
                scratch_affine_calibrator,
            )
            method_predictions[AFFINE_CALIBRATED_SCRATCH_METHOD].append(
                scratch_affine_prediction
            )
            audit["affine_calibrated_transfer"] = {
                "method": AFFINE_CALIBRATED_TRANSFER_METHOD,
                **affine_calibrator,
                "adapter_scope": "target_output_only",
                "validation_prediction_source": "transfer_seed_ensemble_after_validation_selected_residual_blend",
                "holdout_labels_used": False,
                "full_life_min_max_used": False,
            }
            audit["affine_calibrated_scratch"] = {
                "method": AFFINE_CALIBRATED_SCRATCH_METHOD,
                **scratch_affine_calibrator,
                "adapter_scope": "target_output_only",
                "validation_prediction_source": "scratch_seed_ensemble_after_validation_selected_residual_blend",
                "holdout_labels_used": False,
                "full_life_min_max_used": False,
            }

            endpoint_scale = max(float(np.max(train.endpoints)), 1.0)
            huber_validation_prediction = _predict_trend(
                huber_beta,
                validation,
                degree=1,
                endpoint_scale=endpoint_scale,
            )
            huber_test_prediction = method_predictions[CALIBRATED_TREND_FAMILY][0]

            def _select_trend_hybrid_weight(
                method: str,
                prediction: np.ndarray,
            ) -> tuple[float, float, str]:
                if calibrated_transfer_weight_policy == "fixed":
                    weight = float(calibrated_transfer_weight)
                    return weight, rmse(
                        validation.y * validation.scale,
                        _mix_transfer_with_trend(
                            prediction,
                            huber_validation_prediction,
                            transfer_weight=weight,
                        ),
                    ), "predeclared_fixed_weight"
                if bool(inner_group_loo["available"]):
                    selected = inner_group_loo["trend_hybrid"][method]
                    return (
                        float(selected["selected_weight"]),
                        float(selected["inner_macro_raw_rmse"]),
                        "fit_only_inner_group_loo",
                    )
                weight, score = _select_transfer_trend_weight(
                    validation,
                    prediction,
                    huber_validation_prediction,
                    min_weight=CALIBRATED_MIN_TRANSFER_WEIGHT,
                    max_weight=1.0,
                )
                return weight, score, "chronological_validation_fallback"

            selected_transfer_weight, calibrated_validation_rmse, transfer_hybrid_selection = _select_trend_hybrid_weight(
                "trend_residual_transfer", transfer_validation_ensemble
            )
            selected_scratch_weight, scratch_calibrated_validation_rmse, scratch_hybrid_selection = _select_trend_hybrid_weight(
                "trend_residual_scratch", scratch_validation_ensemble
            )
            method_predictions[CALIBRATED_TRANSFER_METHOD].append(
                _mix_transfer_with_trend(
                    transfer_ensemble,
                    huber_test_prediction,
                    transfer_weight=selected_transfer_weight,
                )
            )
            method_predictions[CALIBRATED_SCRATCH_METHOD].append(
                _mix_transfer_with_trend(
                    scratch_ensemble,
                    huber_test_prediction,
                    transfer_weight=selected_scratch_weight,
                )
            )

            if bool(inner_group_loo["available"]):
                transfer_ridge_selection = inner_group_loo["ridge_hybrid"]["trend_residual_transfer"]
                scratch_ridge_selection = inner_group_loo["ridge_hybrid"]["trend_residual_scratch"]
                ridge_transfer_weight = float(transfer_ridge_selection["selected_weight"])
                ridge_calibrated_validation_rmse = float(transfer_ridge_selection["inner_macro_raw_rmse"])
                ridge_scratch_weight = float(scratch_ridge_selection["selected_weight"])
                ridge_scratch_validation_rmse = float(scratch_ridge_selection["inner_macro_raw_rmse"])
                ridge_hybrid_selection = "fit_only_inner_group_loo"
            else:
                ridge_transfer_weight, ridge_calibrated_validation_rmse = _select_transfer_ridge_weight(
                    validation,
                    transfer_validation_ensemble,
                    ridge_validation_prediction,
                    min_weight=CALIBRATED_MIN_TRANSFER_WEIGHT,
                    max_weight=1.0,
                )
                ridge_scratch_weight, ridge_scratch_validation_rmse = _select_transfer_ridge_weight(
                    validation,
                    scratch_validation_ensemble,
                    ridge_validation_prediction,
                    min_weight=CALIBRATED_MIN_TRANSFER_WEIGHT,
                    max_weight=1.0,
                )
                ridge_hybrid_selection = "chronological_validation_fallback"
            method_predictions[RIDGE_CALIBRATED_TRANSFER_METHOD].append(
                _mix_transfer_with_trend(
                    transfer_ensemble,
                    ridge_prediction,
                    transfer_weight=ridge_transfer_weight,
                )
            )
            method_predictions[RIDGE_CALIBRATED_SCRATCH_METHOD].append(
                _mix_transfer_with_trend(
                    scratch_ensemble,
                    ridge_prediction,
                    transfer_weight=ridge_scratch_weight,
                )
            )
            audit["ridge_calibrated_transfer"] = {
                "method": RIDGE_CALIBRATED_TRANSFER_METHOD,
                "source_transfer_weight": float(ridge_transfer_weight),
                "target_ridge_weight": float(1.0 - ridge_transfer_weight),
                "selection": ridge_hybrid_selection,
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "minimum_source_transfer_weight": CALIBRATED_MIN_TRANSFER_WEIGHT,
                "selection_raw_rmse": float(ridge_calibrated_validation_rmse),
                "target_ridge_alpha": float(ridge_alpha),
                "target_ridge_validation_raw_rmse": float(ridge_validation_rmse),
                "holdout_used": False,
            }
            audit["ridge_calibrated_scratch"] = {
                "method": RIDGE_CALIBRATED_SCRATCH_METHOD,
                "source_scratch_weight": float(ridge_scratch_weight),
                "target_ridge_weight": float(1.0 - ridge_scratch_weight),
                "selection": ridge_hybrid_selection,
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "minimum_scratch_weight": CALIBRATED_MIN_TRANSFER_WEIGHT,
                "selection_raw_rmse": float(ridge_scratch_validation_rmse),
                "target_ridge_alpha": float(ridge_alpha),
                "target_ridge_validation_raw_rmse": float(ridge_validation_rmse),
                "holdout_used": False,
            }
            audit["calibrated_transfer"] = {
                "method": CALIBRATED_TRANSFER_METHOD,
                "source_transfer_weight": float(selected_transfer_weight),
                "causal_trend_weight": float(1.0 - selected_transfer_weight),
                "causal_trend_family": CALIBRATED_TREND_FAMILY,
                "selection": transfer_hybrid_selection,
                "weight_policy": calibrated_transfer_weight_policy,
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "minimum_source_transfer_weight": CALIBRATED_MIN_TRANSFER_WEIGHT,
                "selection_raw_rmse": float(calibrated_validation_rmse),
                "holdout_used": False,
            }
            audit["calibrated_scratch"] = {
                "method": CALIBRATED_SCRATCH_METHOD,
                "source_scratch_weight": float(selected_scratch_weight),
                "causal_trend_weight": float(1.0 - selected_scratch_weight),
                "causal_trend_family": CALIBRATED_TREND_FAMILY,
                "selection": scratch_hybrid_selection,
                "weight_policy": calibrated_transfer_weight_policy,
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "minimum_scratch_weight": CALIBRATED_MIN_TRANSFER_WEIGHT,
                "selection_raw_rmse": float(scratch_calibrated_validation_rmse),
                "holdout_used": False,
            }

            # Record the selected prior as its own deterministic comparator.
            selected_prior_raw = prior_test * test.scale
            method_predictions.setdefault("trend_prior", []).append(selected_prior_raw)
            rows.append({
                "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                "method": "trend_prior", **metrics(truth, selected_prior_raw, test.scale, "trend_prior", None),
                "fit_groups": list(split["fit_groups"]), "validation_groups": list(split["validation_groups"]),
                "test_groups": list(split["test_groups"]), "fit_diagnostics": prior_diag,
            })

            for method, values in method_predictions.items():
                aggregate = np.mean(np.stack(values), axis=0)
                ensemble_method = f"{method}_ensemble"
                rows.append({
                    "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                    "method": ensemble_method, **metrics(truth, aggregate, test.scale, ensemble_method, None),
                    "fit_groups": list(split["fit_groups"]), "validation_groups": list(split["validation_groups"]),
                    "test_groups": list(split["test_groups"]),
                })
                for endpoint, actual, predicted in zip(test.endpoints, truth, aggregate):
                    predictions.append({
                        "source": source_name, "n_shots": n_shots, "outer_holdout": holdout,
                        "method": ensemble_method, "target_group": holdout, "endpoint": int(endpoint),
                        "true_rul": float(actual), "prediction": float(predicted),
                    })
    return rows, predictions, audits


def run(source_names: Sequence[str], args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir) if args.temp_dir else output.parent / ".outerloo_v4_temp"
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
        required = [comsol_path, *(source_paths[name] for name in source_names)]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
        archive: SimulationArchive = load_simulation_archive(comsol_path)
        table = archive.aligned[args.target_feature_tier]
        target_feature_count = len(table.feature_names) * (2 if args.target_include_deltas else 1)
        groups = tuple(sorted(set(table.group_ids.tolist()), key=_natural_group))
        if len(groups) != 5:
            raise ValueError(f"COMSOL outer protocol requires five groups, found {groups}")
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
                model, diagnostic = _train_source_residual_model(
                    bundle,
                    target_features=target_feature_count,
                    seed=seed,
                    epochs=source_epochs,
                    device=device,
                )
                source_models[seed] = copy.deepcopy(model).cpu()
                source_training.append({"seed": seed, **diagnostic})
            rows, prediction_rows, audits = _run_outer(
                source_name,
                bundle,
                source_models,
                table,
                seeds=seeds,
                epochs=target_epochs,
                device=device,
                holdouts=holdouts,
                prior_policy=args.target_prior_policy,
                use_context=args.use_context,
                max_blend_weight=args.max_blend_weight,
                blend_selection_policy=args.blend_selection_policy,
                transfer_residual_head=args.transfer_residual_head,
                calibrated_transfer_weight=args.calibrated_transfer_weight,
                calibrated_transfer_weight_policy=args.calibrated_transfer_weight_policy,
                encoder_anchor_weight=args.encoder_anchor_weight,
                transfer_warmup_epochs=args.transfer_warmup_epochs,
                target_include_deltas=args.target_include_deltas,
                target_adapter_alignment_steps=args.target_adapter_alignment_steps,
            )
            all_rows.extend(rows)
            all_predictions.extend(prediction_rows)
            source_reports[source_name] = {
                "manifest": {
                    "source": source_name,
                    "sha256": bundle.sha256,
                    "unit_names": sorted(bundle.units),
                    "source_train_units": list(bundle.train_names),
                    "source_validation_units": list(bundle.validation_names),
                    "feature_names": list(bundle.feature_names),
                    "source_label_scale": bundle.train.scale,
                },
                "source_training": source_training,
                "target_audit_by_outer_fold": audits,
                "source_model_parameter_count": int(sum(parameter.numel() for parameter in source_models[seeds[0]].parameters())),
            }
        preflight = {
            "schema": "femto_ims_to_comsol_outerloo_preflight_v4",
            "inputs_exist": True,
            "comsol_archive": str(comsol_path),
            "comsol_sha256": archive.sha256,
            "comsol_feature_tier": args.target_feature_tier,
            "comsol_groups": list(groups),
            "outer_holdouts": list(holdouts),
            "target_scaler_fit_only_on_fit_groups": True,
            "target_prior_fit_only_on_fit_groups": True,
            "blend_selection": {
                "policy": args.blend_selection_policy,
                "N>=2": "fit-only inner leave-one-outer-fit-group-out OOF; equal-per-group macro RMSE",
                "N=1": "chronological outer-validation fallback because inner group-LOO is impossible",
                "outer_holdout_used": False,
                "outer_validation_used_for_N>=2_weight_selection": False,
            },
            "full_life_min_max_used_for_target_input": False,
            "holdout_labels_used_for_selection": False,
            "holdout_labels_used_for_fit": False,
            "source_names": list(source_names),
            "execution_device": device,
            "cpu_threads": torch.get_num_threads() if device == "cpu" else None,
            "finite_checks": bool(np.all(np.isfinite(table.features)) and np.all(np.isfinite(table.target))),
            "target_model_shape_check": {"target_features": target_feature_count, "sequence_length": TARGET_SEQ_LEN},
            "transfer_warmup_epochs": int(args.transfer_warmup_epochs),
            "transfer_checkpoint_requires_unfrozen_encoder": True,
            "validation_affine_calibration": {
                "method": AFFINE_CALIBRATED_TRANSFER_METHOD,
                "selection": "chronological_validation_only",
                "holdout_labels_used": False,
                "slope_bounds": list(AFFINE_SLOPE_BOUNDS),
                "intercept_fraction_of_train_label_scale": AFFINE_INTERCEPT_FRACTION,
                "identity_fallback": True,
            },
            "target_input_feature_transform": (
                "raw_levels_plus_causal_first_differences"
                if args.target_include_deltas else "raw_levels"
            ),
            "target_adapter_alignment": {
                "enabled": bool(args.target_adapter_alignment_steps),
                "steps": int(args.target_adapter_alignment_steps),
                "source_data": "source train windows only",
                "target_data": "outer fit-group input windows only; inner OOF fits use an inner subset only",
                "target_labels_used": False,
                "validation_or_outer_holdout_used": False,
            },
        }
        protocol = {
            "schema": "femto_ims_to_comsol_outerloo_v4",
            "source_arms": list(source_names),
            "experiment_designation": args.experiment_designation,
            "target_feature_tier": args.target_feature_tier,
            "target_input_feature_transform": (
                "raw_levels_plus_causal_first_differences"
                if args.target_include_deltas else "raw_levels"
            ),
            "target_protocol": "five COMSOL trajectories outer-LOO; deterministic validation rotation; N fit groups",
            "outer_holdouts": list(holdouts),
            "target_adapter_alignment": {
                "enabled": bool(args.target_adapter_alignment_steps),
                "steps": int(args.target_adapter_alignment_steps),
                "objective": "source_target_projected_and_encoded_moment_matching",
                "source_data": "source train windows only",
                "target_data": "outer fit-group input windows only",
                "target_labels_used": False,
                "validation_or_outer_holdout_used": False,
            },
            "n_shots": list(TARGET_N_SHOTS),
            "schedule": {str(n): outer_schedule(groups, n) for n in TARGET_N_SHOTS},
            "target_scaler": "fit on fit_groups only",
            "target_label_scale": "fit on fit_groups only",
            "trend_prior": "causal linear/quadratic/Huber/Ridge prior fitted on fit_groups only; policy is recorded per run",
            "target_prior_policy": args.target_prior_policy,
            "blend_selection": {
                "policy": args.blend_selection_policy,
                "N>=2": "fit-only inner leave-one-outer-fit-group-out OOF; equal-per-group macro RMSE",
                "N=1": "chronological outer-validation fallback because inner group-LOO is impossible",
                "outer_holdout_used": False,
                "outer_validation_used_for_N>=2_weight_selection": False,
                "tie_break": "smaller primary-model weight",
            },
            "transfer_residual_head": bool(args.transfer_residual_head),
            "validation_affine_calibrated_transfer": {
                "method": AFFINE_CALIBRATED_TRANSFER_METHOD,
                "adapter": "bounded affine map from raw transfer prediction to raw target RUL",
                "fit_data": "chronological validation trajectory only",
                "selection": "identity and bounded affine candidates compared on validation only",
                "slope_bounds": list(AFFINE_SLOPE_BOUNDS),
                "intercept_limit": "0.25 * fit-group target label scale",
                "identity_fallback": True,
                "outer_holdout_labels_used": False,
                "full_life_min_max_used": False,
                "reporting": "paired only with its validation-affine scratch counterpart; not interchangeable with raw pure transfer",
            },
            "ridge_calibrated_transfer": {
                "method": RIDGE_CALIBRATED_TRANSFER_METHOD,
                "source_transfer_weight": "selected_per_outer_fold_by_configured_fit_only_policy",
                "target_ridge_weight": "complement_of_selected_transfer_weight",
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "minimum_source_transfer_weight": CALIBRATED_MIN_TRANSFER_WEIGHT,
                "selection": "configured fit-only inner group-LOO OOF for N>=2; chronological validation fallback for N=1; never outer holdout",
                "target_ridge_fit": "fit_groups_only; alpha selected on chronological validation",
                "holdout_used": False,
                "matched_scratch_method": RIDGE_CALIBRATED_SCRATCH_METHOD,
            },
            "calibrated_transfer": {
                "method": "trend_residual_transfer_calibrated",
                "source_transfer_weight": (
                    float(args.calibrated_transfer_weight)
                    if args.calibrated_transfer_weight_policy == "fixed"
                    else "selected_per_outer_fold_by_configured_fit_only_policy"
                ),
                "causal_trend_weight": (
                    float(1.0 - args.calibrated_transfer_weight)
                    if args.calibrated_transfer_weight_policy == "fixed"
                    else "complement_of_selected_transfer_weight"
                ),
                "predeclared_fixed_source_transfer_weight": float(args.calibrated_transfer_weight),
                "causal_trend_family": CALIBRATED_TREND_FAMILY,
                "weight_policy": args.calibrated_transfer_weight_policy,
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "minimum_source_transfer_weight": CALIBRATED_MIN_TRANSFER_WEIGHT,
                "matched_scratch_method": CALIBRATED_SCRATCH_METHOD,
                "selection": (
                    "configured fit-only inner group-LOO OOF for N>=2; chronological validation fallback for N=1; never outer holdout"
                    if args.calibrated_transfer_weight_policy != "fixed"
                    else "predeclared_fixed_weight; not selected on outer holdout"
                ),
                "holdout_used": False,
            },
            "target_context": {
                "enabled": bool(args.use_context),
                "definition": "first observed target feature vector, repeated causally per window",
                "labels_or_full_life_stats_used": False,
            },
            "residual_adaptation": {
                "blend_selection": "configured fit-only inner group-LOO OOF for N>=2; chronological validation fallback for N=1",
                "max_blend_weight": float(args.max_blend_weight),
                "holdout_used": False,
            },
            "main_transfer": "source-residual-pretrained shared encoder plus target projection and residual head; explicit short frozen warmup, low-LR unfreeze and encoder anchor; retained checkpoint is required to be post-unfreeze; residual mixture weights use fit-only inner group-LOO OOF for N>=2 and chronological validation only as the explicit N=1 fallback",
            "transfer_warmup_epochs": int(args.transfer_warmup_epochs),
            "transfer_checkpoint_requires_unfrozen_encoder": True,
            "encoder_anchor_weight": float(args.encoder_anchor_weight),
            "residual_target": "normalized target RUL minus selected causal target trend",
            "source_residual_target": "normalized source RUL minus source-train-only causal source trend",
            "source_sampling": "source residual pretraining uses train-only inverse-window-count loss weights so every source trajectory has equal effective loss mass",
            "frozen_ablation": "pretrained shared encoder frozen; target projection and residual head trainable",
            "scratch": "same trend-residual target architecture, data, epoch budget, early stopping and seeds; every transfer hybrid has the same calibration operation applied to scratch",
            "matched_scratch_controls": {
                "raw_transfer": "trend_residual_scratch",
                "transfer_huber_hybrid": CALIBRATED_SCRATCH_METHOD,
                "transfer_affine_calibration": AFFINE_CALIBRATED_SCRATCH_METHOD,
                "transfer_ridge_hybrid": RIDGE_CALIBRATED_SCRATCH_METHOD,
                "selection_rule": "fixed weights are identical; configured selection uses matched fit-only inner group-LOO OOF grids for N>=2 and the same chronological-validation fallback grids for N=1",
            },
            "ridge": "target-only baseline with validation-selected alpha",
            "full_life_min_max": "never used for target input conversion",
            "proxy_warning": "COMSOL is a reaction-wheel simulation domain, not real flight telemetry",
            "seeds": list(seeds),
            "source_epochs": source_epochs,
            "target_epochs": target_epochs,
        }
        outer_fold_summary = _summarize(all_rows) + _summarize_all_shots(all_rows)
        report = {
            "schema": "femto_ims_to_comsol_outerloo_report_v4",
            "protocol": protocol,
            "preflight": preflight,
            "sources": source_reports,
            "macro_rows": all_rows,
            "outer_fold_summary": outer_fold_summary,
            "elapsed_sec": time.time() - started,
        }
        write_json(output / "PREFLIGHT.json", preflight)
        write_json(output / "PROTOCOL.json", protocol)
        write_json(output / "DATA_MANIFEST.json", {
            "schema": "femto_ims_to_comsol_outerloo_data_manifest_v4",
            "inputs": {
                "comsol": {"path": str(comsol_path), "sha256": archive.sha256, "groups": list(groups)},
                "sources": {name: source_reports[name]["manifest"] for name in source_names},
            },
            "outer_split": protocol["schedule"],
            "holdout_labels_used_for_selection": False,
        })
        write_json(output / "REPORT.json", report)
        with (output / "MACRO.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in all_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(_json_safe(all_rows))
        with (output / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["source", "n_shots", "outer_holdout", "method", "target_group", "endpoint", "true_rul", "prediction"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_predictions)
        with (output / "OUTER_FOLD_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in outer_fold_summary for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(_json_safe(outer_fold_summary))
        write_json(output / "ACCEPTANCE.json", {
            "schema": "femto_ims_to_comsol_outerloo_acceptance_v4",
            "experiment_designation": args.experiment_designation,
            "submission_eligibility": (
                "requires_unseen_outer_holdout_confirmation"
                if args.experiment_designation == "exploratory_post_hoc_replication"
                else "eligible_for_protocol_scoped_reporting"
            ),
            "criterion": "each transfer variant requires >=5% macro RMSE improvement over its matched scratch control, a majority of paired outer-fold wins, and lower macro RMSE than Ridge and the best deterministic trend; a missing matched scratch control makes the result diagnostic",
            "results": outer_fold_summary,
            "positive_transfer_is_not_inferred_from_ridge_or_trend_baseline_advantage": True,
        })
        write_json(output / "SUPERVISOR_STATUS.json", {
            "schema": "femto_ims_to_comsol_outerloo_v4_status",
            "state": "completed",
            "updated_at": time.time(),
            "output": str(output),
            "rows": len(all_rows),
            "elapsed_sec": time.time() - started,
        })
        return {"output": str(output), "sources": list(source_names), "rows": len(all_rows), "elapsed_sec": time.time() - started}
    finally:
        tempfile.tempdir = previous_tempdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("femto", "ims", "both"), default="both")
    parser.add_argument("--femto-zip", default="data/processed/femto_bearing.zip")
    parser.add_argument("--ims-path", default="data/processed/ims_processed")
    parser.add_argument("--comsol-archive", default="data/raw/competition/reaction_wheel_comsol_degradation.zip")
    parser.add_argument(
        "--target-feature-tier",
        choices=("operational", "estimated"),
        default=TARGET_FEATURE_TIER,
        help="Observable COMSOL feature tier; oracle latent features are intentionally excluded from v4.",
    )
    parser.add_argument(
        "--target-include-deltas",
        action="store_true",
        help="Append causal first differences to target windows for all target methods and baselines.",
    )
    parser.add_argument(
        "--target-adapter-alignment-steps",
        type=int,
        default=0,
        help="Unsupervised source-to-target input-adapter moment-matching steps using source train and target fit inputs only.",
    )
    parser.add_argument("--output", default="outputs/femto_ims_to_comsol_outerloo_v4_transfer_enhanced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--source-epochs", type=int, default=SOURCE_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=max(120, TARGET_EPOCHS))
    parser.add_argument(
        "--target-prior-policy",
        choices=PRIOR_POLICIES,
        default="validation",
        help="Target prior selection policy; both choices fit only on outer fit groups.",
    )
    parser.add_argument(
        "--transfer-residual-head",
        action="store_true",
        help="Initialize target residual head from the source residual head as an explicit ablation.",
    )
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Enable first-observation target context as an explicit ablation.",
    )
    parser.add_argument(
        "--max-blend-weight",
        type=float,
        default=1.0,
        help="Maximum learned-residual weight selected by the configured fit-only policy.",
    )
    parser.add_argument(
        "--blend-selection-policy",
        choices=BLEND_SELECTION_POLICIES,
        default="inner_group_loo",
        help=(
            "Use fit-only inner group leave-one-out OOF selection for N>=2, "
            "or retain chronological validation as an explicit ablation."
        ),
    )
    parser.add_argument(
        "--calibrated-transfer-weight",
        type=float,
        default=CALIBRATED_DEFAULT_TRANSFER_WEIGHT,
        help="Predeclared source-transfer weight in the explicit transfer + Huber trend hybrid.",
    )
    parser.add_argument(
        "--calibrated-transfer-weight-policy",
        choices=CALIBRATED_WEIGHT_POLICIES,
        default="validation",
        help=(
            "Use the configured fit-only blend-selection policy for the transfer/trend "
            "mixture, or retain the fixed predeclared weight."
        ),
    )
    parser.add_argument(
        "--experiment-designation",
        choices=("confirmatory", "exploratory_post_hoc_replication"),
        default="confirmatory",
        help="Evidence label recorded in audit artifacts; it never changes model selection or evaluation.",
    )
    parser.add_argument(
        "--encoder-anchor-weight",
        type=float,
        default=0.02,
        help="Penalty keeping the unfrozen transfer encoder near the source encoder; selected before the outer test.",
    )
    parser.add_argument(
        "--transfer-warmup-epochs",
        type=int,
        default=DEFAULT_TRANSFER_WARMUP_EPOCHS,
        help="Fixed head/projection-only epochs before transfer encoder unfreezing; the retained transfer checkpoint must be post-unfreeze.",
    )
    parser.add_argument("--source-max-files", type=int, default=None)
    parser.add_argument("--holdouts", default="")
    parser.add_argument("--temp-dir", default="")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not CALIBRATED_MIN_TRANSFER_WEIGHT <= args.calibrated_transfer_weight <= 1.0:
        parser.error(
            "--calibrated-transfer-weight must be in "
            f"[{CALIBRATED_MIN_TRANSFER_WEIGHT}, 1]"
        )
    if args.encoder_anchor_weight < 0.0:
        parser.error("--encoder-anchor-weight must be non-negative")
    if args.transfer_warmup_epochs < 1:
        parser.error("--transfer-warmup-epochs must be at least 1")
    if args.target_adapter_alignment_steps < 0:
        parser.error("--target-adapter-alignment-steps must be non-negative")
    source_names = ("femto", "ims") if args.source == "both" else (args.source,)
    print(json.dumps(run(source_names, args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
