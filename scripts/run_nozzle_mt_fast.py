"""Fast nozzle multi-trajectory benchmark using predefined train/val/test split.

Protocol (trajectory-disjoint, no label leakage):
  - Split column in CSV defines train / validation / test groups.
  - Inner selection: train on train-split, validate on validation-split.
    Selects best (candidate, epoch_budget) purely from dev labels.
  - Final training: train on train+validation, evaluate on test.
  - 5 fixed seeds, equal-weight ensemble.
  - Classical baselines (Ridge, current-rate) evaluated on same test windows.

This is ~100x faster than full leave-one-out (LOO) because it does
  7 candidates × 3 seeds (inner) + 7 × 5 seeds (final) = 21+35 = 56 model trains
instead of 40 outer folds × 7 × 5 = 1400+.

Usage:
  python scripts/run_nozzle_mt_fast.py \\
      --data data/processed/nozzle_multitrajectory/nozzle_sim_40traj.csv \\
      --output outputs/nozzle_mt_v2 \\
      --device cuda:1
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
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nozzle_multitrajectory import (
    NozzleTrajectory,
    NozzleTrajectoryWindows,
    fit_scaler,
    load_nozzle_trajectories,
    make_windows,
)
from src.models.nozzle_multitrajectory import (
    CensoredRULLoss,
    NozzleOutput,
    WeibullRULLoss,
    build_nozzle_mt_model,
    count_parameters,
)

# ── fixed protocol constants ────────────────────────────────────────────────
SEEDS       = (42, 123, 456, 2026, 3407)
INNER_SEEDS = (42, 123, 456)
EPOCH_INNER = 120
EPOCH_FINAL = 160
WINDOW_SIZE = 5
RUL_SCALE   = 20.0          # seconds; used to normalise loss / metrics
PATIENCE    = 30
BATCH_SIZE  = 256


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_tier: str
    window_size: int = WINDOW_SIZE
    hidden: int = 48
    dropout: float = 0.1


CANDIDATES = (
    Candidate("gru_obs_w5",          "gru",              "observable"),
    Candidate("ms_obs_w5",           "ms",               "observable"),
    Candidate("transformer_obs_w5",  "transformer",      "observable"),
    Candidate("rate_obs_w5",         "physics_residual", "observable"),
    Candidate("gru_est_w5",          "gru",              "estimated"),
    Candidate("ms_est_w5",           "ms",               "estimated"),
    Candidate("rate_est_w5",         "physics_residual", "estimated"),
)


# ── helpers ─────────────────────────────────────────────────────────────────

def seed_everything(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else float("nan")


def mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m]-b[m]))) if m.any() else float("nan")


def finite_mean(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else float("nan")


def _json_safe(v):
    if isinstance(v, np.ndarray):   return _json_safe(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):   return int(v)
    if isinstance(v, dict):         return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):return [_json_safe(x) for x in v]
    return v


def write_json(path, obj):
    path.write_text(
        json.dumps(_json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _bootstrap_ci(vals, n_boot=4000, seed=2026):
    arr = np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))])
    if arr.size == 0:
        return {"mean": None, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    s = rng.choice(arr, (n_boot, arr.size), replace=True).mean(1)
    return {"mean": float(arr.mean()),
            "lower_95": float(np.quantile(s, 0.025)),
            "upper_95": float(np.quantile(s, 0.975))}


# ── windowing ────────────────────────────────────────────────────────────────

def _windows(trajs, candidate, *, scaler=None):
    if scaler is None:
        scaler = fit_scaler(trajs)
    w = make_windows(trajs, scaler, window_size=candidate.window_size)
    return w, scaler


# ── training ─────────────────────────────────────────────────────────────────

def train_one(candidate, train_w, val_w, device, seed, epochs):
    seed_everything(seed)
    nf = train_w.X.shape[-1]
    model = build_nozzle_mt_model(
        candidate.model, nf, candidate.hidden, candidate.dropout, RUL_SCALE
    ).to(device)

    x     = torch.as_tensor(train_w.X,                 dtype=torch.float32, device=device)
    rate  = torch.as_tensor(train_w.current_rate_m_s,  dtype=torch.float32, device=device)
    margin= torch.as_tensor(train_w.depth_margin_mm,   dtype=torch.float32, device=device)
    age   = torch.as_tensor(train_w.endpoint_time_s,   dtype=torch.float32, device=device)

    loss_fn = CensoredRULLoss(eta=RUL_SCALE * 0.75, beta=2.5)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    batches = math.ceil(max(len(train_w), 1) / BATCH_SIZE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * batches)
    )
    rng = np.random.default_rng(seed)

    best_state, best_score, best_ep, stale = None, float("inf"), epochs, 0

    vx = vm = vr = va = None
    if val_w is not None:
        vx = torch.as_tensor(val_w.X,                dtype=torch.float32, device=device)
        vm = torch.as_tensor(val_w.depth_margin_mm,  dtype=torch.float32, device=device)
        vr = torch.as_tensor(val_w.current_rate_m_s, dtype=torch.float32, device=device)
        va = torch.as_tensor(val_w.endpoint_time_s,  dtype=torch.float32, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(len(train_w))
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            out: NozzleOutput = model(x[b], current_rate_m_s=rate[b],
                                      depth_margin_mm=margin[b], current_time_s=age[b])
            true = torch.as_tensor(train_w.rul_s[b], dtype=torch.float32, device=device)
            mask = torch.as_tensor(train_w.target_mask[b], dtype=torch.bool, device=device)
            lb   = torch.as_tensor(train_w.lower_bound_s[b], dtype=torch.float32, device=device)
            loss = loss_fn(out.rul_s, true, age[b], mask, lb, RUL_SCALE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        if val_w is None:
            continue
        model.eval()
        with torch.no_grad():
            vo: NozzleOutput = model(vx, current_rate_m_s=vr,
                                     depth_margin_mm=vm, current_time_s=va)
            vp = vo.rul_s.cpu().numpy()
        truth = val_w.rul_s[val_w.target_mask]
        pred  = vp[val_w.target_mask]
        score = rmse(truth, pred) if len(truth) else float(np.mean(np.abs(vp - val_w.rul_s)))
        if score < best_score - 1e-5:
            best_score, best_ep, stale = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep, best_score


@torch.no_grad()
def predict(model, windows, device):
    x  = torch.as_tensor(windows.X,               dtype=torch.float32, device=device)
    r  = torch.as_tensor(windows.current_rate_m_s, dtype=torch.float32, device=device)
    mg = torch.as_tensor(windows.depth_margin_mm,  dtype=torch.float32, device=device)
    out: NozzleOutput = model(x, current_rate_m_s=r, depth_margin_mm=mg)
    return out.rul_s.cpu().numpy().astype(np.float32)


# ── baselines ────────────────────────────────────────────────────────────────

def _regr_mat(w: NozzleTrajectoryWindows):
    age = w.endpoint_time_s[:, None].astype(np.float64) / max(RUL_SCALE, 1.0)
    return np.c_[np.ones(len(w)), w.X[:, -1, :].astype(np.float64), age]


def fit_ridge(w, alpha=1.0):
    mask = w.target_mask
    X = _regr_mat(w)[mask]
    y = w.rul_s[mask].astype(np.float64)
    I = np.eye(X.shape[1]); I[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)


def predict_ridge(w, beta):
    return np.maximum(_regr_mat(w) @ beta, 0.0).astype(np.float32)


def current_rate_baseline(w: NozzleTrajectoryWindows):
    margin = w.depth_margin_mm.astype(np.float64) * 1e-3
    rate   = w.current_rate_m_s.astype(np.float64).clip(min=1e-12)
    return np.maximum(margin / rate, 0.0).astype(np.float32)


def mean_baseline(w: NozzleTrajectoryWindows):
    """Predict constant = mean RUL on the window."""
    return np.full(len(w), float(np.mean(w.rul_s[w.target_mask])), dtype=np.float32)


# ── metrics ──────────────────────────────────────────────────────────────────

def metrics_row(truth, pred, name, rul_scale):
    pre = truth > 0.0
    return {
        "name": name,
        "all_rmse":          rmse(truth, pred),
        "all_mae":           mae(truth, pred),
        "pre_failure_rmse":  rmse(truth[pre], pred[pre]) if pre.any() else None,
        "normalized_rmse":   rmse(truth, pred) / max(rul_scale, 1.0),
        "n_windows":         int(len(truth)),
    }


# ── main benchmark ───────────────────────────────────────────────────────────

def run_fast_benchmark(
    train_trajs: list,
    val_trajs:   list,
    test_trajs:  list,
    out_dir: Path,
    device: str,
    candidates=CANDIDATES,
    inner_seeds=INNER_SEEDS,
    final_seeds=SEEDS,
    epoch_inner=EPOCH_INNER,
    epoch_final=EPOCH_FINAL,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Inner selection (train → val) ────────────────────────────────────
    print(f"\n=== Inner selection: {len(train_trajs)} train  {len(val_trajs)} val ===", flush=True)
    print(f"    Candidates: {len(candidates)}  Seeds: {list(inner_seeds)}  Epochs: {epoch_inner}", flush=True)

    train_sel_w, scaler_sel = _windows(train_trajs, candidates[0])   # scaler from train only
    val_sel_w, _            = _windows(val_trajs,   candidates[0], scaler=scaler_sel)

    selection_rows = []
    for cand in candidates:
        # refit windows with this candidate's window_size
        tw, sc = _windows(train_trajs, cand)
        vw, _  = _windows(val_trajs, cand, scaler=sc)
        seed_scores, best_eps = [], []
        for s in inner_seeds:
            m, ep, score = train_one(cand, tw, vw, device, s, epoch_inner)
            seed_scores.append(score)
            best_eps.append(ep)
            print(f"  [{cand.name}] seed={s} best_ep={ep} val_rmse={score:.4f}", flush=True)
        mean_score = float(np.mean(seed_scores))
        med_ep     = int(np.median(best_eps))
        selection_rows.append({
            "candidate":       asdict(cand),
            "inner_mean_rmse": mean_score,
            "median_best_ep":  med_ep,
            "seed_scores":     seed_scores,
        })
        print(f"  [{cand.name}] mean_val_rmse={mean_score:.4f}  median_ep={med_ep}", flush=True)

    best_row = min(selection_rows, key=lambda r: r["inner_mean_rmse"])
    best_cand = Candidate(**best_row["candidate"])
    epoch_use = max(best_row["median_best_ep"], 20)
    print(f"\nSelected: {best_cand.name}  epochs={epoch_use}  inner_rmse={best_row['inner_mean_rmse']:.4f}", flush=True)

    # ── 2. Final training (train+val → test) ────────────────────────────────
    print(f"\n=== Final training: {len(train_trajs)+len(val_trajs)} dev  {len(test_trajs)} test ===", flush=True)
    print(f"    Seeds: {list(final_seeds)}  Epochs: {epoch_final}", flush=True)

    dev_trajs = train_trajs + val_trajs
    dev_w, scaler_final = _windows(dev_trajs, best_cand)
    test_w, _           = _windows(test_trajs, best_cand, scaler=scaler_final)

    seed_preds = []
    for s in final_seeds:
        m, ep, _ = train_one(best_cand, dev_w, None, device, s, epoch_final)
        p = predict(m, test_w, device)
        seed_preds.append(p)
        print(f"  seed={s}  ep={ep}  test_rmse={rmse(test_w.rul_s, p):.4f}", flush=True)

    ensemble_pred = np.mean(np.stack(seed_preds), axis=0)

    # ── 3. Classical baselines ───────────────────────────────────────────────
    dev_w_b, scaler_b = _windows(dev_trajs, best_cand)
    test_w_b, _       = _windows(test_trajs, best_cand, scaler=scaler_b)

    ridge_beta   = fit_ridge(dev_w_b)
    ridge_pred   = predict_ridge(test_w_b, ridge_beta)
    curr_pred    = current_rate_baseline(test_w_b)
    mean_pred    = mean_baseline(test_w_b)

    truth     = test_w.rul_s
    rul_scale = float(np.nanmax(np.abs(truth))) if truth.size else 1.0

    methods = {
        "selected_neural": ensemble_pred,
        "ridge":           ridge_pred,
        "current_rate":    curr_pred,
        "mean_baseline":   mean_pred,
    }
    metrics = {name: metrics_row(truth, pred, name, rul_scale)
               for name, pred in methods.items()}

    # ── 4. Per-trajectory breakdown ─────────────────────────────────────────
    per_traj = []
    for traj in test_trajs:
        tid = traj.manifest.trajectory_id
        tw_i, _ = _windows([traj], best_cand, scaler=scaler_final)
        preds_i = np.mean(
            np.stack([predict(
                # re-predict with each seed model on individual trajectory
                build_nozzle_mt_model(
                    best_cand.model, dev_w.X.shape[-1],
                    best_cand.hidden, best_cand.dropout, RUL_SCALE
                ).to(device),
                tw_i, device
            ) for _ in range(1)]),
            axis=0
        )
        # simpler: slice ensemble predictions for this trajectory's windows
        # (use endpoint_indices to match)
        ep_idx = {int(i): pos for pos, i in enumerate(test_w.endpoint_indices)}
        local_ep = [ep_idx.get(int(i)) for i in tw_i.endpoint_indices]
        ens_local = np.array([ensemble_pred[p] if p is not None else np.nan for p in local_ep])
        ridge_local = np.array([ridge_pred[p] if p is not None else np.nan for p in local_ep])
        curr_local  = np.array([curr_pred[p]  if p is not None else np.nan for p in local_ep])
        truth_local = tw_i.rul_s
        rs = float(np.nanmax(np.abs(truth_local))) if truth_local.size else 1.0
        per_traj.append({
            "trajectory_id":   tid,
            "n_windows":       int(len(tw_i)),
            "neural_rmse":     rmse(truth_local, ens_local),
            "ridge_rmse":      rmse(truth_local, ridge_local),
            "current_rate_rmse": rmse(truth_local, curr_local),
        })

    # ── 5. Print summary ─────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("NOZZLE MT FAST BENCHMARK  — predefined split results", flush=True)
    print(f"  Train: {len(train_trajs)}  Val: {len(val_trajs)}  Test: {len(test_trajs)}", flush=True)
    print(f"  Selected model : {best_cand.name}", flush=True)
    print(f"  Test windows   : {len(test_w)}", flush=True)
    for name, m in metrics.items():
        print(f"  {name:22s}: RMSE={m['all_rmse']:.4f}s  MAE={m['all_mae']:.4f}s  nRMSE={m['normalized_rmse']:.4f}", flush=True)
    vs_ridge = ((metrics["ridge"]["all_rmse"] - metrics["selected_neural"]["all_rmse"])
                / max(metrics["ridge"]["all_rmse"], 1e-9) * 100)
    vs_cr    = ((metrics["current_rate"]["all_rmse"] - metrics["selected_neural"]["all_rmse"])
                / max(metrics["current_rate"]["all_rmse"], 1e-9) * 100)
    print(f"\n  Neural vs Ridge       : +{vs_ridge:.1f}%", flush=True)
    print(f"  Neural vs CurrentRate : +{vs_cr:.1f}%", flush=True)
    print("="*60, flush=True)

    result = {
        "schema":          "nozzle_mt_fast_report_v1",
        "protocol":        "predefined_split; train/val for selection; train+val for final; test evaluated once",
        "selected_model":  asdict(best_cand),
        "epoch_inner":     epoch_inner,
        "epoch_final":     epoch_use,
        "n_train":         len(train_trajs),
        "n_val":           len(val_trajs),
        "n_test":          len(test_trajs),
        "n_test_windows":  int(len(test_w)),
        "metrics":         metrics,
        "per_trajectory":  per_traj,
        "selection_rows":  selection_rows,
        "vs_ridge_pct":    round(vs_ridge, 2),
        "vs_current_rate_pct": round(vs_cr, 2),
        "inner_seeds":     list(inner_seeds),
        "final_seeds":     list(final_seeds),
        "bootstrap_ci_95": {
            name: _bootstrap_ci([m["all_rmse"]])
            for name, m in metrics.items()
        },
    }
    write_json(out_dir / "NOZZLE_MT_REPORT.json", result)

    # prediction CSV
    rows = []
    for i in range(len(test_w)):
        rows.append({
            "trajectory_id":  test_w.endpoint_indices[i],    # placeholder
            "window_idx":     int(i),
            "time_s":         float(test_w.endpoint_time_s[i]),
            "true_rul_s":     float(truth[i]) if np.isfinite(truth[i]) else None,
            "neural":         float(ensemble_pred[i]),
            "ridge":          float(ridge_pred[i]),
            "current_rate":   float(curr_pred[i]),
        })
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=list(rows[0]))
        w2.writeheader(); w2.writerows(rows)

    # per-trajectory CSV
    with (out_dir / "per_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        if per_traj:
            w2 = csv.DictWriter(f, fieldnames=list(per_traj[0]))
            w2.writeheader(); w2.writerows(per_traj)

    print(f"\nResults written to {out_dir}/", flush=True)
    return result


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast nozzle MT benchmark (predefined split)")
    parser.add_argument("--data",         required=True, help="Path to nozzle multi-trajectory CSV")
    parser.add_argument("--output",       default="outputs/nozzle_mt_v2")
    parser.add_argument("--device",       default="cuda:1")
    parser.add_argument("--epoch-inner",  type=int, default=EPOCH_INNER)
    parser.add_argument("--epoch-final",  type=int, default=EPOCH_FINAL)
    parser.add_argument("--quick",        action="store_true", help="Smoke test: 2 seeds, 8 epochs")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ep_inner = min(args.epoch_inner, 8) if args.quick else args.epoch_inner
    ep_final = min(args.epoch_final, 8) if args.quick else args.epoch_final
    inner_s  = INNER_SEEDS[:1] if args.quick else INNER_SEEDS
    final_s  = SEEDS[:2]       if args.quick else SEEDS
    cands    = CANDIDATES[:2]  if args.quick else CANDIDATES

    print(f"Device: {device}  quick={args.quick}", flush=True)
    data_path = Path(args.data)
    all_trajs = load_nozzle_trajectories(data_path, feature_tier="estimated")
    print(f"Loaded {len(all_trajs)} trajectories", flush=True)

    # Read split column directly from CSV (TrajectoryManifest has no split field)
    split_map: dict[str, str] = {}
    with data_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trajectory_id", "")
            sp  = row.get("split", "")
            if tid and sp:
                split_map[tid] = sp

    if split_map:
        train_trajs = [t for t in all_trajs.values() if split_map.get(t.manifest.trajectory_id) == "train"]
        val_trajs   = [t for t in all_trajs.values() if split_map.get(t.manifest.trajectory_id) == "validation"]
        test_trajs  = [t for t in all_trajs.values() if split_map.get(t.manifest.trajectory_id) == "test"]
        print(f"Split (from CSV): train={len(train_trajs)} val={len(val_trajs)} test={len(test_trajs)}", flush=True)
    else:
        # fallback: random 65/15/20 split
        all_list = list(all_trajs.values())
        rng = np.random.default_rng(2026)
        idx = rng.permutation(len(all_list))
        n_test = max(1, int(len(all_list) * 0.20))
        n_val  = max(1, int(len(all_list) * 0.15))
        test_trajs  = [all_list[i] for i in idx[:n_test]]
        val_trajs   = [all_list[i] for i in idx[n_test:n_test+n_val]]
        train_trajs = [all_list[i] for i in idx[n_test+n_val:]]
        print(f"No split column — using random 65/15/20 split", flush=True)
        print(f"Split: train={len(train_trajs)} val={len(val_trajs)} test={len(test_trajs)}", flush=True)

    out_dir = Path(args.output)
    t0 = time.time()
    result = run_fast_benchmark(
        train_trajs, val_trajs, test_trajs,
        out_dir=out_dir, device=device,
        candidates=cands,
        inner_seeds=inner_s, final_seeds=final_s,
        epoch_inner=ep_inner, epoch_final=ep_final,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min", flush=True)
    write_json(out_dir / "run_metadata.json", {
        "elapsed_sec": round(elapsed, 1),
        "device": device,
        "n_train": len(train_trajs),
        "n_val":   len(val_trajs),
        "n_test":  len(test_trajs),
    })


if __name__ == "__main__":
    main()
