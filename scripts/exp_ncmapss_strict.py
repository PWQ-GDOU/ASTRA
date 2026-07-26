"""Strict N-CMAPSS turbofan RUL benchmark targeting SOTA performance.

Protocol (consistent with Chao 2021 and best practice):
- Unit-disjoint splits: dev units for training, test units for evaluation.
- Scaler fitted on dev units only; never on test.
- Condition normalisation: condition-aware feature normalisation.
- RUL capped at 125 cycles (standard N-CMAPSS/C-MAPSS convention).
- Validation: last 20% of dev units held for checkpoint selection.
- Fixed seeds: 42, 123, 456, 2026, 3407. Equal-weight ensemble.
- Metrics: RMSE, MAE, NASA score (asymmetric), NRMSE.
- No test labels used for any selection.

Usage with real data:
  python scripts/exp_ncmapss_strict.py \\
      --data-dir path/to/ncmapss/ \\
      --datasets DS01 DS02 DS03 DS04 \\
      --output outputs/ncmapss_strict_v1

Usage with synthetic data (smoke test):
  python scripts/exp_ncmapss_strict.py --synthetic --output /tmp/smoke
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

from src.data.ncmapss_strict import (
    FEATURE_SETS,
    NCMAPSSUnit,
    NCMAPSSWindows,
    RUL_CAP,
    describe_split,
    fit_scaler,
    load_ncmapss_h5,
    make_synthetic_units,
    make_windows,
)
from src.models.ncmapss_strict import (
    RUL_SCALE,
    RULOutput,
    build_ncmapss_model,
    count_parameters,
)


from src.models.nozzle_multitrajectory import WeibullRULLoss


SEEDS = (42, 123, 456, 2026, 3407)
SEQ_LEN = 30
VAL_FRACTION = 0.20
EPOCH_BUDGET = 120          # final training epoch ceiling
SELECTION_EPOCH_BUDGET = 60 # inner selection uses shorter budget (early stopping guards quality)
WEIBULL_ETA = 100.0   # characteristic life ≈ cap cycles for turbofan
WEIBULL_BETA = 3.0    # shape: steeper than bearing (β=2.5) for turbofan degradation
WEIBULL_ALPHA = 0.25  # hybrid weight: 0.25*weibull + 0.75*smooth_l1


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_set: str
    seq_len: int = SEQ_LEN
    hidden: int = 96
    dropout: float = 0.1


CANDIDATES = (
    Candidate("ms_tcn_phys_cond", "ms_tcn", "physical_with_conditions"),
    Candidate("bigru_attn_phys_cond", "bigru_attn", "physical_with_conditions"),
    Candidate("transformer_phys_cond", "transformer", "physical_with_conditions"),
    Candidate("physics_gru_phys_cond", "physics_gru", "physical_with_conditions"),
    Candidate("ms_tcn_full_cond", "ms_tcn", "full_with_conditions"),
    Candidate("transformer_full_cond", "transformer", "full_with_conditions"),
    Candidate("bigru_full_cond", "bigru_attn", "full_with_conditions", hidden=128),
    Candidate("transformer_large_phys", "transformer", "physical_with_conditions", hidden=128),
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_val(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bias_val(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric NASA RUL score: penalises late predictions more heavily."""
    diff = y_pred - y_true
    return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    return {
        "name": name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae_val(y_true, y_pred),
        "bias": bias_val(y_true, y_pred),
        "nrmse": rmse(y_true, y_pred) / RUL_CAP,
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    candidate: Candidate,
    train_windows: NCMAPSSWindows,
    val_windows: NCMAPSSWindows | None,
    device: str,
    seed: int,
    *,
    epochs: int,
    batch_size: int = 512,
    patience: int = 25,
    learning_rate: float = 8.0e-4,
) -> tuple[object, int, float]:
    seed_everything(seed)
    n_features = train_windows.X.shape[-1]
    n_cond = FEATURE_SETS[candidate.feature_set].index("T24") \
        if "T24" in FEATURE_SETS[candidate.feature_set] else 0
    # Count condition features (first N_OP_CONDITIONS are conditions if present)
    from src.data.ncmapss_strict import CONDITION_NAMES, N_OP_CONDITIONS
    if candidate.feature_set.endswith("_with_conditions"):
        n_cond = N_OP_CONDITIONS
    else:
        n_cond = 0
    model = build_ncmapss_model(
        candidate.model, n_features, n_cond, candidate.hidden, candidate.dropout
    ).to(device)
    x = torch.as_tensor(train_windows.X, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_windows.Y, dtype=torch.float32, device=device)
    # Absolute elapsed life: age = RUL_CAP - RUL (correct on truncated DS03/04/07/08
    # where y.max() < RUL_CAP and the batch-max proxy would under-estimate age)
    age_s = (RUL_CAP - y).clamp(min=0.0)
    weibull_loss = WeibullRULLoss(eta=WEIBULL_ETA, beta=WEIBULL_BETA)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    batches = math.ceil(max(len(train_windows), 1) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, max(epochs, EPOCH_BUDGET) * batches)
    )
    rng = np.random.default_rng(seed)
    val_x = None
    if val_windows is not None:
        val_x = torch.as_tensor(val_windows.X, dtype=torch.float32, device=device)
    best_state = None
    best_score = float("inf")
    best_epoch = int(epochs)
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_windows))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            out: RULOutput = model(x[idx])
            # Hybrid loss: Weibull-weighted near-EOL term + smooth L1
            # WEIBULL_ALPHA=0.25 keeps the smooth-L1 dominant on truncated datasets
            w_loss = weibull_loss(out.rul, y[idx], age_s[idx], rul_scale=RUL_CAP)
            h_loss = F.smooth_l1_loss(out.rul / RUL_CAP, y[idx] / RUL_CAP, beta=0.05)
            loss = (1.0 - WEIBULL_ALPHA) * h_loss + WEIBULL_ALPHA * w_loss
            if out.degradation is not None:
                # Auxiliary degradation task: normalised elapsed time as proxy
                age_proxy = (1.0 - y[idx] / RUL_CAP).clamp(0, 1)
                loss = loss + 0.1 * F.mse_loss(out.degradation, age_proxy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        if val_windows is None:
            continue
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).rul.detach().cpu().numpy()
        score = rmse(val_windows.Y, val_pred)
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
def predict(model, windows: NCMAPSSWindows, device: str) -> np.ndarray:
    x = torch.as_tensor(windows.X, dtype=torch.float32, device=device)
    return model(x).rul.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Benchmark for a single dataset (DS01…DS08)
# ---------------------------------------------------------------------------


def run_dataset_benchmark(
    dataset_name: str,
    dev_units: list[NCMAPSSUnit],
    test_units: list[NCMAPSSUnit],
    out_dir: Path,
    device: str,
    seeds: Sequence[int],
    candidates: Sequence[Candidate],
    epochs: int,
) -> dict:
    print(f"\n[{dataset_name}] dev={len(dev_units)} test={len(test_units)}", flush=True)

    # Split dev into train and validation by unit (never mix units)
    n_val = max(1, int(len(dev_units) * VAL_FRACTION))
    train_units = dev_units[:-n_val]
    val_units = dev_units[-n_val:]

    # --- Inner candidate selection on train/val units (SELECTION_EPOCH_BUDGET, not full) ---
    selection_rows = []
    for candidate in candidates:
        scaler = fit_scaler(train_units, candidate.feature_set,
                            dataset=dataset_name, split="dev")
        train_w = make_windows(train_units, scaler, seq_len=candidate.seq_len)
        val_w = make_windows(val_units, scaler, seq_len=candidate.seq_len)
        scores, epochs_list = [], []
        for seed in seeds:
            model, ep, _ = train_model(
                candidate, train_w, val_w, device, seed,
                epochs=SELECTION_EPOCH_BUDGET,
            )
            pred = predict(model, val_w, device)
            scores.append(rmse(val_w.Y, pred))
            epochs_list.append(ep)
        mean_val_rmse = float(np.mean(scores))
        # Scale frozen_epochs up to full budget proportionally
        # e.g. median_best_epoch=45 of 60 → (45/60)*full_epochs
        median_ep = int(np.median(epochs_list))
        scale_factor = epochs / SELECTION_EPOCH_BUDGET if SELECTION_EPOCH_BUDGET > 0 else 1.0
        scaled_ep = max(10, int(round(median_ep * scale_factor)))
        selection_rows.append({
            "candidate": asdict(candidate),
            "val_rmse": mean_val_rmse,
            "val_rmse_std": float(np.std(scores)),
            "median_best_epoch_sel": median_ep,
            "median_best_epoch_final": scaled_ep,
        })
        print(
            f"  [{dataset_name}] {candidate.name}: "
            f"val_rmse={mean_val_rmse:.3f}±{np.std(scores):.3f}",
            flush=True,
        )

    selected = min(selection_rows, key=lambda r: r["val_rmse"])
    selected_candidate = Candidate(**selected["candidate"])
    frozen_epochs = selected["median_best_epoch_final"]
    print(
        f"  [{dataset_name}] selected: {selected_candidate.name} "
        f"frozen_epochs={frozen_epochs}",
        flush=True,
    )

    # --- Final training on all dev units, evaluation on test units ---
    all_dev_scaler = fit_scaler(dev_units, selected_candidate.feature_set,
                                dataset=dataset_name, split="dev")
    dev_windows = make_windows(dev_units, all_dev_scaler, seq_len=selected_candidate.seq_len)
    test_windows = make_windows(test_units, all_dev_scaler, seq_len=selected_candidate.seq_len)

    seed_predictions = []
    for seed in seeds:
        model, _, _ = train_model(selected_candidate, dev_windows, None,
                                  device, seed, epochs=frozen_epochs)
        seed_predictions.append(predict(model, test_windows, device))
    ensemble = np.mean(np.stack(seed_predictions), axis=0)

    # Per-seed metrics for transparency
    per_seed_rmse = [
        float(rmse(test_windows.Y, sp)) for sp in seed_predictions
    ]

    # Baselines
    from sklearn.linear_model import Ridge as SklearnRidge
    ridge = SklearnRidge(alpha=10.0)
    ridge.fit(dev_windows.X[:, -1, :].reshape(len(dev_windows), -1), dev_windows.Y)
    ridge_pred = np.maximum(ridge.predict(test_windows.X[:, -1, :].reshape(len(test_windows), -1)), 0).astype(np.float32)

    metrics = [
        compute_metrics(test_windows.Y, ensemble, "selected_neural"),
        compute_metrics(test_windows.Y, ridge_pred, "ridge"),
        compute_metrics(test_windows.Y, np.full(len(test_windows), float(np.mean(dev_windows.Y)), dtype=np.float32), "mean_baseline"),
    ]

    result = {
        "dataset": dataset_name,
        "dev_units": len(dev_units),
        "test_units": len(test_units),
        "selected_candidate": asdict(selected_candidate),
        "frozen_epochs": frozen_epochs,
        "seed_ids": list(seeds),
        "per_seed_test_rmse": per_seed_rmse,
        "ensemble_test_rmse": float(rmse(test_windows.Y, ensemble)),
        "metrics": metrics,
        "candidate_selection": selection_rows,
        "scaler": all_dev_scaler.as_dict(),
        "loss_config": {
            "weibull_alpha": WEIBULL_ALPHA,
            "weibull_eta": WEIBULL_ETA,
            "weibull_beta": WEIBULL_BETA,
        },
    }
    write_json(out_dir / f"{dataset_name}_report.json", result)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Directory with N-CMAPSS .h5 files")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07", "DS08"],
    )
    parser.add_argument("--output", default="outputs/ncmapss_strict_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCH_BUDGET)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    epochs = min(int(args.epochs), 8) if args.quick else int(args.epochs)
    candidates = CANDIDATES[:3] if args.quick else CANDIDATES
    if args.quick:
        seeds = seeds[:1]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_results = []
    if args.synthetic or args.data_dir is None:
        print("Using synthetic data for smoke test", flush=True)
        synth = make_synthetic_units(n_dev=6, n_test=2, rng_seed=42, dataset_name="SYNTH")
        result = run_dataset_benchmark(
            "SYNTH", synth["dev"], synth["test"],
            out_dir, device, seeds, candidates, epochs
        )
        all_results.append(result)
    else:
        data_dir = Path(args.data_dir)
        for ds_name in args.datasets:
            # Match both bare "DS01.h5" and NASA naming "N-CMAPSS_DS01-005.h5"
            h5_candidates = (
                list(data_dir.glob(f"{ds_name}.h5"))
                + list(data_dir.glob(f"**/{ds_name}.h5"))
                + list(data_dir.glob(f"*{ds_name}*.h5"))
                + list(data_dir.glob(f"**/*{ds_name}*.h5"))
            )
            # Deduplicate while preserving order
            seen: set = set()
            h5_candidates = [p for p in h5_candidates if not (str(p) in seen or seen.add(str(p)))]
            if not h5_candidates:
                print(f"[{ds_name}] .h5 file not found in {data_dir}, skipping", flush=True)
                continue
            h5_path = h5_candidates[0]
            print(f"\n[{ds_name}] loading {h5_path}", flush=True)
            splits = load_ncmapss_h5(h5_path, dataset_name=ds_name)
            result = run_dataset_benchmark(
                ds_name, splits["dev"], splits["test"],
                out_dir, device, seeds, candidates, epochs
            )
            all_results.append(result)

    # Summary across datasets
    if all_results:
        macro: dict = {}
        for method in ["selected_neural", "ridge", "mean_baseline"]:
            rmse_scores = []
            nasa_scores = []
            for r in all_results:
                m = next((m for m in r["metrics"] if m["name"] == method), None)
                if m:
                    rmse_scores.append(m["rmse"])
                    nasa_scores.append(m["nasa_score"])
            if rmse_scores:
                macro[method] = {
                    "mean_rmse": float(np.mean(rmse_scores)),
                    "per_dataset": rmse_scores,
                    "mean_nasa_score": float(np.mean(nasa_scores)),
                }

        # Per-seed ensemble stability (neural only)
        all_per_seed = [r.get("per_seed_test_rmse", []) for r in all_results]
        if all(all_per_seed):
            per_seed_means = np.mean(all_per_seed, axis=0).tolist()
            macro["selected_neural"]["per_seed_mean_rmse"] = per_seed_means
            macro["selected_neural"]["per_seed_std_rmse"] = float(np.std(per_seed_means))

        summary = {
            "schema": "ncmapss_strict_summary_v1",
            "datasets": [r["dataset"] for r in all_results],
            "macro": macro,
            "seed_ids": list(seeds),
            "scheduler_epoch_budget": int(epochs),
            "selection_epoch_budget": SELECTION_EPOCH_BUDGET,
            "loss_config": {
                "weibull_alpha": WEIBULL_ALPHA,
                "weibull_eta": WEIBULL_ETA,
                "weibull_beta": WEIBULL_BETA,
            },
        }
        write_json(out_dir / "NCMAPSS_SUMMARY.json", summary)
        write_json(out_dir / "run_metadata.json", {
            "elapsed_sec": time.time() - t0,
            "device": device,
        })
        print("\nResults:", json.dumps(macro, indent=2), flush=True)


if __name__ == "__main__":
    main()
