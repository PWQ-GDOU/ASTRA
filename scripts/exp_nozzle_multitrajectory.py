"""Grouped multi-trajectory nozzle ablation RUL benchmark.

Protocol:
- Outer split: leave-one-trajectory-out (or condition-held-out when available).
- Inner split: rotating inner LOO within development trajectories for
  candidate/hyperparameter selection.
- Scaler, epoch budget, feature tier, and model family all selected
  exclusively on training/development trajectories.
- Test trajectories are evaluated exactly once after selection is frozen.
- Five fixed seeds; equal-weight ensemble; group bootstrap 95% CI.

Usage (with real data):
  python scripts/exp_nozzle_multitrajectory.py \
      --data path/to/multi_trajectory.csv \
      --output outputs/nozzle_multitrajectory_v1

Usage (synthetic smoke-test):
  python scripts/exp_nozzle_multitrajectory.py --synthetic --output /tmp/smoke
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

from src.data.nozzle_multitrajectory import (
    NozzleTrajectory,
    NozzleTrajectoryWindows,
    describe_trajectory,
    fit_scaler,
    load_nozzle_trajectories,
    make_windows,
    sha256_file,
)
from src.models.nozzle_multitrajectory import (
    CensoredRULLoss,
    NozzleOutput,
    WeibullRULLoss,
    build_nozzle_mt_model,
    count_parameters,
)


SEEDS = (42, 123, 456, 2026, 3407)
EPOCH_BUDGET = 160
WINDOW_SIZE = 5
RUL_SCALE = 20.0

FEATURE_TIERS = ("observable", "estimated")


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_tier: str
    window_size: int = WINDOW_SIZE
    hidden: int = 48
    dropout: float = 0.1


CANDIDATES = (
    Candidate("gru_obs_w5", "gru", "observable"),
    Candidate("ms_obs_w5", "ms", "observable"),
    Candidate("transformer_obs_w5", "transformer", "observable"),
    Candidate("rate_obs_w5", "physics_residual", "observable"),
    Candidate("gru_est_w5", "gru", "estimated"),
    Candidate("ms_est_w5", "ms", "estimated"),
    Candidate("rate_est_w5", "physics_residual", "estimated"),
)


# ---------------------------------------------------------------------------
# Utilities
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if np.any(mask) else float("nan")


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if np.any(mask) else float("nan")


def bias_val(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(b[mask] - a[mask])) if np.any(mask) else float("nan")


def finite_mean(values: Iterable) -> float:
    nums = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(nums)) if nums else float("nan")


def _json_safe(v):
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, Mapping):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256_string(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _metrics(truth: np.ndarray, pred: np.ndarray, name: str, rul_scale: float) -> dict:
    pre_mask = truth > 0.0
    return {
        "name": name,
        "all_rmse": rmse(truth, pred),
        "all_mae": mae(truth, pred),
        "all_bias": bias_val(truth, pred),
        "pre_failure_rmse": rmse(truth[pre_mask], pred[pre_mask]) if np.any(pre_mask) else None,
        "pre_failure_mae": mae(truth[pre_mask], pred[pre_mask]) if np.any(pre_mask) else None,
        "normalized_rmse": rmse(truth, pred) / max(rul_scale, 1.0),
        "n_all": int(len(truth)),
        "n_pre_failure": int(np.sum(pre_mask)),
    }


def _bootstrap_ci(values: Sequence[float], seed: int = 2026, n_boot: int = 4000) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _windows(
    trajectories: Sequence[NozzleTrajectory],
    candidate: Candidate,
    *,
    scaler=None,
) -> tuple[NozzleTrajectoryWindows, object]:
    if scaler is None:
        scaler = fit_scaler(trajectories)
    windows = make_windows(trajectories, scaler, window_size=candidate.window_size)
    return windows, scaler


def train_model(
    candidate: Candidate,
    train_windows: NozzleTrajectoryWindows,
    val_windows: NozzleTrajectoryWindows | None,
    device: str,
    seed: int,
    *,
    epochs: int,
    batch_size: int = 256,
    patience: int = 30,
) -> tuple[object, int, float]:
    seed_everything(seed)
    n_features = train_windows.X.shape[-1]
    model = build_nozzle_mt_model(
        candidate.model, n_features, candidate.hidden, candidate.dropout, RUL_SCALE
    ).to(device)

    x = torch.as_tensor(train_windows.X, dtype=torch.float32, device=device)
    rate = torch.as_tensor(train_windows.current_rate_m_s, dtype=torch.float32, device=device)
    margin = torch.as_tensor(train_windows.depth_margin_mm, dtype=torch.float32, device=device)
    age = torch.as_tensor(train_windows.endpoint_time_s, dtype=torch.float32, device=device)

    loss_fn = CensoredRULLoss(eta=RUL_SCALE * 0.75, beta=2.5)
    opt = torch.optim.AdamW(model.parameters(), lr=7.0e-4, weight_decay=1.0e-4)
    batches = math.ceil(max(len(train_windows), 1) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, max(epochs, EPOCH_BUDGET) * batches)
    )
    rng = np.random.default_rng(seed)

    best_state = None
    best_score = float("inf")
    best_epoch = int(epochs)
    stale = 0

    val_x = val_rate = val_margin = val_age = None
    if val_windows is not None:
        val_x = torch.as_tensor(val_windows.X, dtype=torch.float32, device=device)
        val_rate = torch.as_tensor(val_windows.current_rate_m_s, dtype=torch.float32, device=device)
        val_margin = torch.as_tensor(val_windows.depth_margin_mm, dtype=torch.float32, device=device)
        val_age = torch.as_tensor(val_windows.endpoint_time_s, dtype=torch.float32, device=device)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_windows))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            out: NozzleOutput = model(
                x[idx],
                current_rate_m_s=rate[idx],
                depth_margin_mm=margin[idx],
                current_time_s=age[idx],
            )
            true = torch.as_tensor(train_windows.rul_s[idx], dtype=torch.float32, device=device)
            mask = torch.as_tensor(train_windows.target_mask[idx], dtype=torch.bool, device=device)
            lb = torch.as_tensor(train_windows.lower_bound_s[idx], dtype=torch.float32, device=device)
            loss = loss_fn(out.rul_s, true, age[idx], mask, lb, RUL_SCALE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

        if val_windows is None:
            continue

        model.eval()
        with torch.no_grad():
            val_out: NozzleOutput = model(val_x, current_rate_m_s=val_rate, depth_margin_mm=val_margin)
            val_pred = val_out.rul_s.detach().cpu().numpy()
        truth = val_windows.rul_s[val_windows.target_mask]
        pred = val_pred[val_windows.target_mask]
        score = rmse(truth, pred) if len(truth) else float(np.mean(np.abs(val_pred - val_windows.rul_s)))
        if score < best_score - 1.0e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, float(best_score)


@torch.no_grad()
def predict(model, windows: NozzleTrajectoryWindows, device: str) -> np.ndarray:
    x = torch.as_tensor(windows.X, dtype=torch.float32, device=device)
    rate = torch.as_tensor(windows.current_rate_m_s, dtype=torch.float32, device=device)
    margin = torch.as_tensor(windows.depth_margin_mm, dtype=torch.float32, device=device)
    out: NozzleOutput = model(x, current_rate_m_s=rate, depth_margin_mm=margin)
    return out.rul_s.detach().cpu().numpy().astype(np.float32)


def _train_ensemble(
    candidate: Candidate,
    train_trajectories: Sequence[NozzleTrajectory],
    test_trajectories: Sequence[NozzleTrajectory],
    device: str,
    seeds: Sequence[int],
    epochs: int,
) -> tuple[dict, NozzleTrajectoryWindows, NozzleTrajectoryWindows]:
    train_windows, scaler = _windows(train_trajectories, candidate)
    test_windows, _ = _windows(test_trajectories, candidate, scaler=scaler)
    predictions = []
    for seed in seeds:
        model, _, _ = train_model(candidate, train_windows, None, device, seed, epochs=epochs)
        predictions.append(predict(model, test_windows, device))
    matrix = np.stack(predictions, axis=0)
    return {
        "candidate": asdict(candidate),
        "scaler": scaler.as_dict(),
        "seed_ids": list(seeds),
        "ensemble": np.mean(matrix, axis=0),
        "seed_predictions": matrix,
        "parameter_count": count_parameters(
            build_nozzle_mt_model(candidate.model, train_windows.X.shape[-1], candidate.hidden, candidate.dropout)
        ),
    }, train_windows, test_windows


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def _regr_matrix(windows: NozzleTrajectoryWindows) -> np.ndarray:
    age = windows.endpoint_time_s.astype(np.float64)[:, None] / max(RUL_SCALE, 1.0)
    return np.c_[np.ones(len(windows)), windows.X[:, -1, :].astype(np.float64), age]


def fit_ridge(windows: NozzleTrajectoryWindows, alpha: float = 1.0) -> np.ndarray:
    mask = windows.target_mask
    x = _regr_matrix(windows)[mask]
    y = windows.rul_s[mask].astype(np.float64)
    ident = np.eye(x.shape[1])
    ident[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * ident, x.T @ y)


def predict_ridge(windows: NozzleTrajectoryWindows, beta: np.ndarray) -> np.ndarray:
    return np.maximum(_regr_matrix(windows) @ beta, 0.0).astype(np.float32)


def current_rate_baseline(windows: NozzleTrajectoryWindows) -> np.ndarray:
    margin = windows.depth_margin_mm.astype(np.float64) * 1.0e-3
    rate = windows.current_rate_m_s.astype(np.float64).clip(min=1.0e-12)
    return np.maximum(margin / rate, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Inner selection
# ---------------------------------------------------------------------------


def select_candidate(
    all_trajectories: Sequence[NozzleTrajectory],
    holdout_id: str,
    candidates: Sequence[Candidate],
    device: str,
    seeds: Sequence[int],
    epochs: int,
    inner_val_fraction: float = 0.0,
) -> dict:
    """Select best candidate for the given outer holdout.

    When ``inner_val_fraction > 0`` a fixed random subset of development
    trajectories is used as the inner validation set (fast mode).
    When ``inner_val_fraction == 0`` full rotating inner LOO is used (slow,
    ≈39× more training runs).
    """
    development = [t for t in all_trajectories if t.manifest.trajectory_id != holdout_id]
    rows = []
    # --- Build inner train/val split once for all candidates ---
    if inner_val_fraction > 0.0 and len(development) > 1:
        rng_inner = np.random.default_rng(42)
        n_val = max(1, int(len(development) * inner_val_fraction))
        perm = rng_inner.permutation(len(development))
        inner_val = [development[i] for i in perm[:n_val]]
        inner_fit = [development[i] for i in perm[n_val:]] or development
        inner_folds = [(inner_fit, inner_val)]   # single fixed split
    else:
        # Full rotating inner LOO
        inner_folds = [
            ([t for t in development if t.manifest.trajectory_id != val.manifest.trajectory_id]
             or development,
             [val])
            for val in development
        ]
    for candidate in candidates:
        scores, epoch_list = [], []
        for fit, val_list in inner_folds:
            fw, scaler = _windows(fit, candidate)
            vw, _ = _windows(val_list, candidate, scaler=scaler)
            seed_preds = []
            for seed in seeds:
                model, ep, _ = train_model(candidate, fw, vw, device, seed, epochs=epochs)
                seed_preds.append(predict(model, vw, device))
                epoch_list.append(ep)
            pred = np.mean(np.stack(seed_preds), axis=0)
            truth = vw.rul_s[vw.target_mask]
            if len(truth):
                scores.append(rmse(truth, pred[vw.target_mask]))
        rows.append({
            "candidate": asdict(candidate),
            "inner_mean_rmse": float(np.mean(scores)) if scores else float("nan"),
            "median_best_epoch": int(np.median(epoch_list)) if epoch_list else int(epochs),
            "n_inner_trajectories": len(development),
        })
    selected = min(rows, key=lambda r: r["inner_mean_rmse"])
    return {
        "outer_holdout": holdout_id,
        "selected": selected["candidate"],
        "fixed_epochs": selected["median_best_epoch"],
        "candidate_rows": rows,
        "development_trajectory_ids": [t.manifest.trajectory_id for t in development],
    }


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    trajectories: Sequence[NozzleTrajectory],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
    candidates: Sequence[Candidate],
    epochs: int,
    inner_val_fraction: float = 0.20,
) -> dict:
    selections = []
    for holdout in [t.manifest.trajectory_id for t in trajectories]:
        print(f"[select] outer={holdout}", flush=True)
        selections.append(select_candidate(
            trajectories, holdout, candidates, device, seeds, epochs,
            inner_val_fraction=inner_val_fraction,
        ))

    config = {
        "schema": "nozzle_multitrajectory_config_v1",
        "protocol": "grouped LOO; outer=trajectory, inner=trajectory; no test labels in selection",
        "seed_ids": list(seeds),
        "scheduler_epoch_budget": int(epochs),
        "candidates": [asdict(c) for c in candidates],
        "window_size": WINDOW_SIZE,
        "rul_scale_s": RUL_SCALE,
        "outer_selections": selections,
    }
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2).encode()
    config["config_sha256"] = hashlib.sha256(payload).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_config.json", config)

    sel_by_id = {s["outer_holdout"]: s for s in selections}
    folds, prediction_rows, per_seed_rows = [], [], []

    for test_traj in trajectories:
        holdout_id = test_traj.manifest.trajectory_id
        sel = sel_by_id[holdout_id]
        train_traj = [t for t in trajectories if t.manifest.trajectory_id != holdout_id]
        candidate = Candidate(**sel["selected"])
        epochs_final = int(sel["fixed_epochs"])
        print(f"[final] outer={holdout_id} candidate={candidate.name} epochs={epochs_final}", flush=True)

        info, train_w, test_w = _train_ensemble(candidate, train_traj, [test_traj], device, seeds, epochs_final)

        # Classical baselines
        train_ridge = fit_ridge(train_w)
        ridge_pred = predict_ridge(test_w, train_ridge)
        current_rate_pred = current_rate_baseline(test_w)

        truth = test_w.rul_s
        rul_scale_traj = max(float(truth.max()), 1.0)

        metrics = [
            _metrics(truth, info["ensemble"], "selected_neural", rul_scale_traj),
            _metrics(truth, ridge_pred, "ridge", rul_scale_traj),
            _metrics(truth, current_rate_pred, "current_rate", rul_scale_traj),
        ]

        fold = {
            "holdout_id": holdout_id,
            "status": test_traj.manifest.status,
            "candidate": asdict(candidate),
            "epochs": epochs_final,
            "parameter_count": info["parameter_count"],
            "metrics": metrics,
            "scaler": info["scaler"],
        }
        folds.append(fold)

        for seed, seed_pred in zip(info["seed_ids"], info["seed_predictions"]):
            m = _metrics(truth, seed_pred, f"seed_{seed}", rul_scale_traj)
            per_seed_rows.append({"holdout_id": holdout_id, "candidate": candidate.name, "seed": int(seed), **m})

        for i in range(len(test_w)):
            prediction_rows.append({
                "holdout_id": holdout_id,
                "endpoint": int(test_w.endpoint_indices[i]),
                "time_s": float(test_w.endpoint_time_s[i]),
                "true_rul_s": float(truth[i]) if np.isfinite(truth[i]) else None,
                "selected_neural": float(info["ensemble"][i]),
                "ridge": float(ridge_pred[i]),
                "current_rate": float(current_rate_pred[i]),
            })

    methods = ["selected_neural", "ridge", "current_rate"]
    macro = {
        m: {
            "all_rmse": finite_mean([
                next(r for r in fold["metrics"] if r["name"] == m)["all_rmse"]
                for fold in folds if fold["status"] == "event_observed"
            ]),
            "pre_failure_rmse": finite_mean([
                next(r for r in fold["metrics"] if r["name"] == m)["pre_failure_rmse"]
                for fold in folds if fold["status"] == "event_observed"
            ]),
        }
        for m in methods
    }

    event_folds = [f for f in folds if f["status"] == "event_observed"]
    ci = {
        m: _bootstrap_ci([
            next(r for r in fold["metrics"] if r["name"] == m)["all_rmse"]
            for fold in event_folds
        ])
        for m in methods
    }

    result = {
        "schema": "nozzle_multitrajectory_report_v1",
        "protocol": "grouped trajectory LOO; event-observed folds in macro",
        "n_trajectories": len(trajectories),
        "n_event_observed": len(event_folds),
        "folds": folds,
        "macro_metrics": macro,
        "bootstrap_ci_95": ci,
        "seed_ids": list(seeds),
        "frozen_config_sha256": config["config_sha256"],
    }
    write_json(out_dir / "NOZZLE_MT_REPORT.json", result)

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        if prediction_rows:
            w = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
            w.writeheader(); w.writerows(prediction_rows)

    with (out_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        if per_seed_rows:
            w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0]))
            w.writeheader(); w.writerows(per_seed_rows)

    with (out_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [{"holdout_id": fold["holdout_id"], "status": fold["status"], **m} for fold in folds for m in fold["metrics"]]
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    print(json.dumps(macro, indent=2), flush=True)
    return result


# ---------------------------------------------------------------------------
# Synthetic trajectory generator (smoke test only)
# ---------------------------------------------------------------------------


def _synthetic_trajectories(n: int = 6) -> list[NozzleTrajectory]:
    """Generate synthetic multi-trajectory data for local testing without real CSV."""
    from src.data.nozzle_multitrajectory import (
        NozzleTrajectory, TrajectoryManifest,
        NozzleTrajectoryScaler, fit_scaler,
    )
    import io, csv as csv_mod

    lines = ["trajectory_id,condition_id,time_s,cumulative_ablation_depth_mm,"
             "ablation_rate_m_s,failure_depth_mm,solid_temperature_K,"
             "heat_flux_W_m2,pressure_Pa"]
    for i in range(n):
        tid = f"T{i+1:02d}"
        cid = f"C{(i % 3) + 1}"
        threshold = 0.25 + 0.01 * i
        t_vals = np.linspace(0, 18 + i * 0.5, 30)
        depth = threshold * (t_vals / t_vals[-1]) ** (0.8 + 0.02 * i)
        rate = np.gradient(depth, t_vals).clip(min=1e-6)
        temp = 800 + 50 * i + 30 * t_vals / t_vals[-1]
        flux = 5e5 + 1e4 * i
        pressure = 1.4e6
        for j, t in enumerate(t_vals):
            lines.append(f"{tid},{cid},{t:.4f},{depth[j]:.6f},{rate[j]:.6f},"
                         f"{threshold:.4f},{temp[j]:.2f},{flux:.0f},{pressure:.0f}")

    text = "\n".join(lines) + "\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        tmp = fh.name
    return list(load_nozzle_trajectories(tmp, feature_tier="estimated").values())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to multi-trajectory CSV")
    parser.add_argument("--output", default="outputs/nozzle_multitrajectory_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCH_BUDGET)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke test")
    parser.add_argument(
        "--inner-val-fraction", type=float, default=0.20,
        help="Fraction of dev trajectories used as fixed inner validation (0=full LOO, default 0.20)",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    epochs = min(int(args.epochs), 8) if args.quick else int(args.epochs)
    candidates = CANDIDATES
    if args.quick:
        seeds = seeds[:1]
        candidates = CANDIDATES[:3]

    if args.synthetic or args.data is None:
        print("Using synthetic trajectories for smoke test", flush=True)
        trajectories = _synthetic_trajectories(n=4 if args.quick else 6)
    else:
        data_path = Path(args.data)
        trajectories = list(load_nozzle_trajectories(data_path, feature_tier="estimated").values())
        print(f"Loaded {len(trajectories)} trajectories from {data_path}", flush=True)

    out_dir = Path(args.output)
    t0 = time.time()
    result = run_benchmark(trajectories, out_dir, device, seeds, candidates, epochs,
                           inner_val_fraction=args.inner_val_fraction)
    write_json(out_dir / "run_metadata.json", {
        "elapsed_sec": time.time() - t0,
        "device": device,
        "n_trajectories": len(trajectories),
        "seed_ids": list(seeds),
    })


if __name__ == "__main__":
    main()
