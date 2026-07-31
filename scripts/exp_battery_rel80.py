"""Battery RUL benchmark — rel80 protocol (80 % SOH threshold), expanded NASA pool.

Protocol:
  - EOL = first cycle where capacity <= 0.80 * initial_capacity  (rel80)
  - Outer LOOCV on the four "primary" cells: B0005, B0006, B0007, B0018
      (B0007 is now event-observed under rel80, unlike strict14)
  - Inner candidate selection on remaining primary cells only (cell-disjoint)
  - Training pool: primary cells + B0040/B0042/B0043/B0044/B0046/B0047/B0048
      (all event-observed, 31-46 cycles)  +
      right-censored aux cells with censored-loss contribution only
  - Scaler fit on training cells; never touches test cell
  - 5 fixed seeds, equal-weight ensemble
  - Metrics: RMSE / MAE / MAPE / nRMSE-100 / bias  per cell and macro

Usage:
  python scripts/exp_battery_rel80.py \
      --data-dir data/processed/nasa_battery/5. Battery Data Set \
      --output outputs/battery_rel80_v1 \
      --device cuda:1 --epochs 300
"""
from __future__ import annotations
import argparse, csv, json, math, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.battery_strict import (
    FEATURE_NAMES, BatterySeries, BatteryWindowSet,
    fit_scaler, load_battery_mat, make_windows, materialize_battery,
)
from src.models.battery_strict import RUL_SCALE, build_battery_model, count_parameters

# ── protocol constants ───────────────────────────────────────────────────────
PROTOCOL      = "rel80"
PRIMARY_CELLS = ("B0005", "B0006", "B0007", "B0018")   # LOOCV target
AUX_EVENT_CELLS = (
    "B0042", "B0043", "B0044", "B0046", "B0047", "B0048",  # ~1.5-1.7Ah initial, similar chemistry
)
AUX_CENSORED_CELLS = (   # short right-censored cells — censored loss only, no exact RUL
    "B0025", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032",
)
MIN_CYCLES    = 15   # skip cells with < MIN_CYCLES observed cycles
SEEDS         = (42, 123, 456, 2026, 3407)
NRMSE_CYCLES  = 100.0
COMMON_EP     = 23   # common endpoint for windowed evaluation (same as strict14)

@dataclass(frozen=True)
class Candidate:
    name: str; model: str; seq_len: int
    use_age: bool = True; use_capacity_aux: bool = True

CANDIDATES = (
    Candidate("life_l16",         "life",        16),
    Candidate("life_l24",         "life",        24),
    Candidate("gru_l16",          "gru",         16),
    Candidate("ms_l16",           "ms",          16),
    Candidate("transformer_l16",  "transformer", 16),
    Candidate("life_l16_no_aux",  "life",        16, use_capacity_aux=False),
    Candidate("life_l16_no_age",  "life",        16, use_age=False),
)

# ── helpers ──────────────────────────────────────────────────────────────────
def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def rmse(a, b):
    m = np.isfinite(a)&np.isfinite(b)
    return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else float("nan")

def mae(a, b):
    m = np.isfinite(a)&np.isfinite(b)
    return float(np.mean(np.abs(a[m]-b[m]))) if m.any() else float("nan")

def mape(a, b):
    m = np.isfinite(a)&np.isfinite(b)&(np.abs(a)>1e-3)
    return float(np.mean(np.abs((a[m]-b[m])/a[m]))*100) if m.any() else float("nan")

def finite_mean(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else float("nan")

def write_json(path, obj):
    def _s(v):
        if isinstance(v, np.ndarray): return _s(v.tolist())
        if isinstance(v, (np.floating,float)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, dict): return {str(k):_s(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [_s(x) for x in v]
        return v
    path.write_text(json.dumps(_s(obj), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")

# ── training ─────────────────────────────────────────────────────────────────
def _age_tensor(window_set, device):
    """Normalised endpoint index used as age feature (same as exp_battery_strict.py)."""
    return torch.as_tensor(
        window_set.endpoints.astype(np.float32) / RUL_SCALE,
        dtype=torch.float32, device=device,
    )

def train_model(candidate, train_set, val_set, device, seed, epochs, patience=40):
    seed_everything(seed)
    model = build_battery_model(
        candidate.model,
        n_features=train_set.X.shape[-1],
        use_age=candidate.use_age,
        use_capacity_aux=candidate.use_capacity_aux,
    ).to(device)
    x  = torch.as_tensor(train_set.X,              dtype=torch.float32, device=device)
    y  = torch.as_tensor(train_set.rul,             dtype=torch.float32, device=device)
    m  = torch.as_tensor(train_set.target_mask,     dtype=torch.bool,    device=device)
    lb = torch.as_tensor(train_set.lower_bound_rul, dtype=torch.float32, device=device)
    d5 = torch.as_tensor(train_set.future_delta_5,  dtype=torch.float32, device=device)
    d10= torch.as_tensor(train_set.future_delta_10, dtype=torch.float32, device=device)
    m5 = torch.as_tensor(train_set.future_mask_5,   dtype=torch.bool,    device=device)
    m10= torch.as_tensor(train_set.future_mask_10,  dtype=torch.bool,    device=device)
    age= _age_tensor(train_set, device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    batches = math.ceil(max(len(train_set),1)/2048)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs*batches))
    rng = np.random.default_rng(seed)
    best_state, best_score, stale = None, float("inf"), 0

    vx = vy = None
    if val_set is not None:
        vx  = torch.as_tensor(val_set.X, dtype=torch.float32, device=device)
        vage= _age_tensor(val_set, device)
        vy  = val_set.rul[val_set.target_mask]

    for ep in range(1, epochs+1):
        model.train()
        idx = rng.permutation(len(train_set))
        for s in range(0, len(idx), 2048):
            b = idx[s:s+2048]
            out = model(x[b], age[b] if candidate.use_age else None)
            pred = out.rul.squeeze()  # already in cycles
            exact_loss = F.smooth_l1_loss(pred[m[b]], y[b][m[b]], beta=0.5)
            censor_loss = torch.clamp(lb[b][~m[b]] - pred[~m[b]], min=0.).mean() \
                          if (~m[b]).any() else torch.tensor(0., device=device)
            total = exact_loss + 0.5 * censor_loss
            if candidate.use_capacity_aux:
                if m5[b].any():
                    total = total + 0.15*F.smooth_l1_loss(
                        out.capacity_delta_5[m5[b]], d5[b][m5[b]], beta=0.02)
                if m10[b].any():
                    total = total + 0.15*F.smooth_l1_loss(
                        out.capacity_delta_10[m10[b]], d10[b][m10[b]], beta=0.02)
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        if val_set is None: continue
        model.eval()
        with torch.no_grad():
            vout = model(vx, vage if candidate.use_age else None)
            # out.rul is already in cycles (model multiplies by RUL_SCALE internally)
            vp = vout.rul.squeeze().cpu().numpy()[val_set.target_mask]
        score = rmse(vy, vp)
        if score < best_score - 1e-5:
            best_score, stale = score, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience: break

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model, best_score

@torch.no_grad()
def predict(model, window_set, device):
    x   = torch.as_tensor(window_set.X, dtype=torch.float32, device=device)
    age = _age_tensor(window_set, device)
    out = model(x, age)
    # out.rul is already in cycles (model applies RUL_SCALE internally)
    return out.rul.squeeze().cpu().numpy().astype(np.float32)

# ── classical baselines ───────────────────────────────────────────────────────
def _regr_mat(ws):
    feat = ws.X[:, -1, :].astype(np.float64)
    age  = (ws.endpoints.astype(np.float64) / RUL_SCALE)[:, None]
    return np.c_[np.ones(len(ws)), feat, age]

def fit_ridge(ws, alpha=1.0):
    mask = ws.target_mask; X = _regr_mat(ws)[mask]; y = ws.rul[mask]
    I = np.eye(X.shape[1]); I[0,0] = 0.
    return np.linalg.solve(X.T@X + alpha*I, X.T@y)

def predict_ridge(ws, beta):
    return np.maximum(_regr_mat(ws)@beta, 0.).astype(np.float32)

def metrics_row(truth, pred, name):
    pre = truth > 0.
    return {
        "name":     name,
        "rmse":     rmse(truth, pred),
        "mae":      mae(truth, pred),
        "mape":     mape(truth[pre], pred[pre]) if pre.any() else None,
        "bias":     float(np.nanmean(pred - truth)),
        "nrmse_100": rmse(truth, pred) / NRMSE_CYCLES,
        "n":        int(np.sum(np.isfinite(truth))),
    }

# ── data loading ─────────────────────────────────────────────────────────────
def load_cells(data_dir, names, protocol=PROTOCOL):
    cells = {}
    for name in names:
        path = Path(data_dir) / f"{name}.mat"
        if not path.exists():
            print(f"  [skip] {name}: file not found", flush=True)
            continue
        try:
            raw = load_battery_mat(str(path))
            s = materialize_battery(raw, protocol)
            if len(s) < MIN_CYCLES:
                print(f"  [skip] {name}: only {len(s)} cycles", flush=True)
                continue
            cells[name] = s
            print(f"  {name}: {len(s)} cycles  status={s.status}  eol={s.eol_ah:.3f}Ah", flush=True)
        except Exception as e:
            print(f"  [skip] {name}: {e}", flush=True)
    return cells

# ── benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark(all_cells, out_dir, device, epochs=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = {n: s for n, s in all_cells.items() if n in PRIMARY_CELLS}
    event_primary = {n: s for n, s in primary.items() if s.status == "event_observed"}

    print(f"\nPrimary cells: {list(primary.keys())}", flush=True)
    print(f"Event-observed primary: {list(event_primary.keys())}", flush=True)
    print(f"Total training pool: {len(all_cells)} cells", flush=True)

    folds, prediction_rows = [], []

    for holdout_name in PRIMARY_CELLS:
        if holdout_name not in primary:
            print(f"\n[skip] {holdout_name} not available", flush=True)
            continue

        test_series = primary[holdout_name]
        # Training pool: all non-holdout cells with ≥ MIN_CYCLES
        train_series = [s for n, s in all_cells.items() if n != holdout_name]
        # Inner val: other primary event-observed cells (cell-disjoint)
        inner_val_series = [s for n, s in primary.items()
                            if n != holdout_name and s.status == "event_observed"]
        print(f"\n=== Holdout: {holdout_name} ({test_series.status}, {len(test_series)} cycles) ===",
              flush=True)
        print(f"  Train pool: {len(train_series)} cells  |  Inner val: {len(inner_val_series)} cells",
              flush=True)

        # ── Inner candidate selection ─────────────────────────────────────
        feat_names = tuple(FEATURE_NAMES)
        scaler = fit_scaler(train_series, feat_names)
        inner_train_series = [s for s in train_series
                              if s.name not in [v.name for v in inner_val_series]]
        best_cand, best_inner, best_ep = None, float("inf"), 100
        sel_rows = []
        for cand in CANDIDATES:
            tw, _ = make_windows(inner_train_series, cand.seq_len, scaler=scaler,
                                  min_endpoint=COMMON_EP)
            vw, _ = make_windows(inner_val_series, cand.seq_len, scaler=scaler,
                                  min_endpoint=COMMON_EP) if inner_val_series else (None, None)
            scores = []
            for s in SEEDS[:3]:
                _, sc = train_model(cand, tw, vw, device, s, min(epochs, 80), patience=25)
                scores.append(sc)
            mean_sc = float(np.mean(scores))
            print(f"  [{cand.name}] inner_rmse={mean_sc:.2f}", flush=True)
            sel_rows.append({"name": cand.name, "inner_rmse": mean_sc})
            if mean_sc < best_inner:
                best_inner, best_cand = mean_sc, cand
        print(f"  => selected: {best_cand.name}  inner_rmse={best_inner:.2f}", flush=True)

        # ── Final training (full train pool → test) ───────────────────────
        scaler_final = fit_scaler(train_series, tuple(FEATURE_NAMES))
        train_w, _ = make_windows(train_series,  best_cand.seq_len, scaler=scaler_final,
                                   min_endpoint=COMMON_EP)
        test_w,  _ = make_windows([test_series], best_cand.seq_len, scaler=scaler_final,
                                   min_endpoint=COMMON_EP)
        seed_preds = []
        for s in SEEDS:
            m, _ = train_model(best_cand, train_w, None, device, s, epochs)
            p = predict(m, test_w, device)
            seed_preds.append(p)
        ensemble = np.mean(np.stack(seed_preds), axis=0)

        # ── Baselines ─────────────────────────────────────────────────────
        ridge_beta = fit_ridge(train_w)
        ridge_pred = predict_ridge(test_w, ridge_beta)
        truth = test_w.rul

        fold_metrics = [
            metrics_row(truth, ensemble,  "selected_neural"),
            metrics_row(truth, ridge_pred, "ridge"),
        ]
        for m_row in fold_metrics:
            print(f"  {m_row['name']:<22}: RMSE={m_row['rmse']:.3f}  MAE={m_row['mae']:.3f}"
                  f"  MAPE={m_row['mape']:.1f}%  nRMSE={m_row['nrmse_100']:.4f}", flush=True)

        folds.append({
            "holdout": holdout_name, "status": test_series.status,
            "n_cycles": len(test_series), "eol_ah": test_series.eol_ah,
            "life_cycle": test_series.life_cycle,
            "selected_candidate": asdict(best_cand),
            "inner_rmse": best_inner,
            "selection_rows": sel_rows,
            "metrics": fold_metrics,
            "per_seed_rmse": [float(rmse(truth, p)) for p in seed_preds],
        })
        for i in range(len(test_w)):
            prediction_rows.append({
                "holdout": holdout_name,
                "cycle_idx": int(test_w.endpoints[i]),
                "true_rul": float(truth[i]) if np.isfinite(truth[i]) else None,
                "neural": float(ensemble[i]),
                "ridge":  float(ridge_pred[i]),
            })

    # ── Macro metrics ─────────────────────────────────────────────────────
    event_folds = [f for f in folds if f["status"] == "event_observed"]
    def macro(method):
        return {
            "mean_rmse": finite_mean(
                next(m for m in f["metrics"] if m["name"]==method)["rmse"]
                for f in event_folds),
            "mean_mae":  finite_mean(
                next(m for m in f["metrics"] if m["name"]==method)["mae"]
                for f in event_folds),
            "mean_mape": finite_mean(
                next(m for m in f["metrics"] if m["name"]==method)["mape"]
                for f in event_folds),
            "mean_nrmse_100": finite_mean(
                next(m for m in f["metrics"] if m["name"]==method)["nrmse_100"]
                for f in event_folds),
            "per_cell_rmse": [
                {"cell": f["holdout"],
                 "rmse": next(m for m in f["metrics"] if m["name"]==method)["rmse"]}
                for f in event_folds],
        }

    result = {
        "schema": "battery_rel80_v1",
        "protocol": "rel80 (80% SOH); cell-LOOCV; expanded training pool",
        "primary_cells": list(PRIMARY_CELLS),
        "n_event_folds": len(event_folds),
        "macro": {m: macro(m) for m in ["selected_neural", "ridge"]},
        "folds": folds,
        "training_pool_size": len(all_cells),
        "seeds": list(SEEDS),
    }
    write_json(out_dir / "BATTERY_REL80_REPORT.json", result)

    # Print summary
    print("\n" + "="*60, flush=True)
    print("BATTERY rel80 BENCHMARK SUMMARY", flush=True)
    print("="*60, flush=True)
    for method in ["selected_neural", "ridge"]:
        m = result["macro"][method]
        print(f"  {method:<22}: RMSE={m['mean_rmse']:.3f}  MAE={m['mean_mae']:.3f}"
              f"  MAPE={m['mean_mape']:.1f}%  nRMSE/100={m['mean_nrmse_100']:.4f}", flush=True)
    print("="*60, flush=True)

    with (out_dir/"predictions.csv").open("w", newline="", encoding="utf-8") as f:
        if prediction_rows:
            w2 = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
            w2.writeheader(); w2.writerows(prediction_rows)

    return result

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/nasa_battery/5. Battery Data Set")
    parser.add_argument("--output",   default="outputs/battery_rel80_v1")
    parser.add_argument("--device",   default="cuda:1")
    parser.add_argument("--epochs",   type=int, default=300)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    epochs = 8 if args.quick else args.epochs

    print(f"Protocol: {PROTOCOL}  Device: {device}  Epochs: {epochs}", flush=True)

    # Load all cells
    print("\nLoading primary cells...", flush=True)
    primary_cells = load_cells(args.data_dir, PRIMARY_CELLS)
    print("\nLoading aux event cells...", flush=True)
    aux_event     = load_cells(args.data_dir, AUX_EVENT_CELLS)
    print("\nLoading aux censored cells...", flush=True)
    aux_censored  = load_cells(args.data_dir, AUX_CENSORED_CELLS)

    all_cells = {**primary_cells, **aux_event, **aux_censored}
    print(f"\nTotal usable cells: {len(all_cells)}", flush=True)

    if args.quick:
        all_cells = {k: v for k, v in all_cells.items()
                     if k in list(PRIMARY_CELLS[:2]) + list(AUX_EVENT_CELLS[:2])}
        print(f"Quick mode: using {len(all_cells)} cells", flush=True)

    out_dir = Path(args.output)
    t0 = time.time()
    run_benchmark(all_cells, out_dir, device, epochs=epochs)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)
    write_json(out_dir / "run_metadata.json", {
        "elapsed_sec": round(time.time()-t0, 1),
        "device": device, "epochs": epochs,
        "protocol": PROTOCOL,
        "training_pool": list(all_cells.keys()),
    })

if __name__ == "__main__":
    main()
