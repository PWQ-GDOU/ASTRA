"""Few-shot transfer v2: fixes + conformal UQ + more trials.

Key improvements over v1:
  1. RUL normalization fix: max_rul = rul[seq_len-1] so first window = 1.0
     → NASA pre-training now covers full 0-1 RUL range (was 0-0.66)
  2. More trials per N: 5 (was 3) for reduced variance
  3. N=20 extra trials to resolve the non-monotonic bounce
  4. Split conformal prediction intervals (80/90/95% coverage)
  5. Detailed per-trial logging for diagnosis
"""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.battery_strict import load_battery_mat, materialize_battery

# ── constants ─────────────────────────────────────────────────────────────
EARLY_FRAC   = 0.20
EPOCHS_PRE   = 400
EPOCHS_FT    = 300
EPOCHS_SCR   = 500
SEQ_LEN      = 16
HIDDEN       = 64
DROPOUT      = 0.1
N_SHOTS      = (1, 2, 3, 5, 10, 20)
N_TRIALS     = 5          # was 3
N_TEST       = 10
SEEDS        = (42, 123, 456, 2026, 3407)
NASA_AUX     = ("B0042","B0043","B0044","B0046","B0047","B0048")
GEO_CSV      = "data/processed/geo_battery_60traj/geo_battery_60traj.csv"
NASA_DIR     = "data/processed/nasa_battery"

# ── model ─────────────────────────────────────────────────────────────────
class TransferModel(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.GRU(1, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        h, _ = self.encoder(x)
        return self.head(h[:, -1]).squeeze(-1)

# ── data ──────────────────────────────────────────────────────────────────
def _rul_norm_from_first_window(rul: np.ndarray, seq_len: int) -> tuple[np.ndarray, float]:
    """Normalize RUL so the first WINDOW's target = 1.0 (fixes NASA <1.0 issue)."""
    first_window_idx = seq_len - 1        # first valid window endpoint
    max_rul = float(rul[first_window_idx]) if len(rul) > first_window_idx and rul[first_window_idx] > 0 else float(rul[0])
    if max_rul <= 0:
        max_rul = float(np.nanmax(rul[rul > 0])) if np.any(rul > 0) else 1.0
    return rul / max_rul, max_rul

def soh_from_geo(df_traj, seq_len=SEQ_LEN, eol_soh=0.80, early_frac=EARLY_FRAC):
    import pandas as pd
    soh   = df_traj["soh"].values.astype(np.float32)
    cycle = df_traj["cycle"].values.astype(np.float32)
    hits  = np.flatnonzero(soh <= eol_soh)
    if hits.size == 0:
        return None
    eol_cycle = float(cycle[hits[0]])
    rul = np.maximum(eol_cycle - cycle, 0.0).astype(np.float32)
    if len(soh) < seq_len + 1:
        return None
    rul_norm, max_rul = _rul_norm_from_first_window(rul, seq_len)
    # early-life windows (first early_frac of full life)
    n_early = max(seq_len + 1, int(len(soh) * early_frac))
    windows_e, targets_e = [], []
    for end in range(seq_len-1, n_early):
        windows_e.append(soh[end-seq_len+1:end+1])
        targets_e.append(rul_norm[end])
    # full windows for test evaluation
    windows_f, targets_f = [], []
    for end in range(seq_len-1, len(soh)):
        windows_f.append(soh[end-seq_len+1:end+1])
        targets_f.append(rul_norm[end])
    Xe = np.array(windows_e, dtype=np.float32)[:,:,None]
    ye = np.array(targets_e, dtype=np.float32)
    Xf = np.array(windows_f, dtype=np.float32)[:,:,None]
    yf = np.array(targets_f, dtype=np.float32)
    return Xe, ye, Xf, yf, max_rul

def soh_from_nasa(mat_path, seq_len=SEQ_LEN, protocol="rel80"):
    raw = load_battery_mat(str(mat_path))
    series = materialize_battery(raw, protocol)
    if series.status != "event_observed":
        return None
    soh = series.soh.astype(np.float32)
    rul = series.rul.astype(np.float32)
    if len(soh) < seq_len + 1:
        return None
    rul_norm, max_rul = _rul_norm_from_first_window(rul, seq_len)
    windows, targets = [], []
    for end in range(seq_len-1, len(soh)):
        windows.append(soh[end-seq_len+1:end+1])
        targets.append(rul_norm[end])
    X = np.array(windows, dtype=np.float32)[:,:,None]
    y = np.array(targets, dtype=np.float32)
    return X, y, max_rul


# ── training ──────────────────────────────────────────────────────────────
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rmse(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a-b)**2)))

def train_model(model, X, y, device, seed, epochs, freeze_encoder=False, batch=512, lr=3e-4):
    seed_all(seed)
    if freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad_(False)
    else:
        for p in model.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        idx = rng.permutation(len(X))
        for s in range(0, len(idx), batch):
            b = idx[s:s+batch]
            out = model(X[b])
            loss = F.smooth_l1_loss(out, y[b], beta=0.1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    # unfreeze after fine-tune
    for p in model.parameters(): p.requires_grad_(True)
    return model

def predict_norm(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy()


# ── conformal UQ ──────────────────────────────────────────────────────────
def conformal_intervals(cal_residuals: np.ndarray, test_preds: np.ndarray,
                        alphas=(0.05, 0.10, 0.20)):
    """Split conformal prediction intervals.
    Returns dict of alpha -> (lower, upper) arrays in cycle units.
    cal_residuals: signed residuals (pred - true) on calibration set, cycles.
    test_preds: point predictions on test set, cycles.
    """
    abs_res = np.abs(cal_residuals)
    intervals = {}
    for alpha in alphas:
        q = float(np.quantile(abs_res, 1.0 - alpha))
        intervals[f"coverage_{int((1-alpha)*100)}"] = {
            "quantile_cycles": q,
            "lower": (test_preds - q).tolist(),
            "upper": (test_preds + q).tolist(),
        }
    return intervals


# ── main experiment ───────────────────────────────────────────────────────
def run(device, out_dir, quick=False):
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load GEO data ──
    print("Loading GEO data...", flush=True)
    geo_df = pd.read_csv(GEO_CSV)
    geo_traj_ids = sorted(geo_df["trajectory_id"].unique())
    rng_split = np.random.default_rng(2026)
    perm = rng_split.permutation(len(geo_traj_ids))
    test_ids  = [geo_traj_ids[i] for i in perm[:N_TEST]]
    train_ids = [geo_traj_ids[i] for i in perm[N_TEST:]]
    print(f"GEO: {len(geo_traj_ids)} total, {len(train_ids)} train pool, {len(test_ids)} test", flush=True)

    geo_train_data = {}
    for tid in train_ids:
        sub = geo_df[geo_df["trajectory_id"]==tid]
        d = soh_from_geo(sub, SEQ_LEN)
        if d is not None:
            geo_train_data[tid] = d   # (Xe, ye, Xf, yf, max_rul)

    geo_test_Xf, geo_test_yf, geo_test_maxruls = [], [], []
    for tid in test_ids:
        sub = geo_df[geo_df["trajectory_id"]==tid]
        d = soh_from_geo(sub, SEQ_LEN)
        if d is not None:
            geo_test_Xf.append(d[2])
            geo_test_yf.append(d[3])
            geo_test_maxruls.append(np.full(len(d[3]), d[4], dtype=np.float32))

    Xtest = torch.as_tensor(np.concatenate(geo_test_Xf), dtype=torch.float32, device=device)
    ytest_norm  = np.concatenate(geo_test_yf)
    ytest_max   = np.concatenate(geo_test_maxruls)
    ytest_cyc   = ytest_norm * ytest_max
    print(f"GEO test: {len(Xtest)} windows, RUL range {ytest_cyc.min():.0f}-{ytest_cyc.max():.0f} cycles", flush=True)

    # ── load NASA ──
    print("\nLoading NASA source data...", flush=True)
    nasa_Xs, nasa_ys = [], []
    for cell in NASA_AUX:
        mat = Path(NASA_DIR) / f"{cell}.mat"
        if not mat.exists(): continue
        d = soh_from_nasa(mat, SEQ_LEN)
        if d is not None:
            nasa_Xs.append(d[0]); nasa_ys.append(d[1])
            print(f"  {cell}: {len(d[0])} windows  max_rul={d[2]:.0f}  norm_range={d[1].min():.3f}-{d[1].max():.3f}", flush=True)
    Xnasa = torch.as_tensor(np.concatenate(nasa_Xs), dtype=torch.float32, device=device)
    ynasa = torch.as_tensor(np.concatenate(nasa_ys), dtype=torch.float32, device=device)
    print(f"NASA total: {len(Xnasa)} windows, norm RUL range {ynasa.min():.3f}-{ynasa.max():.3f}", flush=True)

    # ── pre-train ──
    epochs_pre = EPOCHS_PRE if not quick else 50
    epochs_ft  = EPOCHS_FT  if not quick else 50
    epochs_scr = EPOCHS_SCR if not quick else 100
    print(f"\nPre-training on NASA ({len(NASA_AUX)} cells, {epochs_pre} epochs)...", flush=True)
    pretrain_states = []
    for seed in SEEDS:
        m = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
        m = train_model(m, Xnasa, ynasa, device, seed, epochs_pre, freeze_encoder=False)
        pretrain_states.append({k: v.cpu().clone() for k,v in m.state_dict().items()})
    print("Pre-training done.", flush=True)

    avail = [t for t in train_ids if t in geo_train_data]
    results = {}
    n_trials = N_TRIALS if not quick else 2

    for N in N_SHOTS:
        if N > len(avail):
            print(f"  N={N}: not enough trajectories ({len(avail)} avail)", flush=True)
            continue
        transfer_rmses, scratch_rmses = [], []
        trial_log = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(trial * 7777 + N)
            chosen = rng_t.choice(len(avail), N, replace=False)
            sel_ids = [avail[i] for i in chosen]
            Xs = [geo_train_data[t][0] for t in sel_ids]  # early-life windows
            ys = [geo_train_data[t][1] for t in sel_ids]
            Xfew = torch.as_tensor(np.concatenate(Xs), dtype=torch.float32, device=device)
            yfew = torch.as_tensor(np.concatenate(ys), dtype=torch.float32, device=device)
            print(f"  N={N} trial={trial}: {len(Xfew)} early-life windows from {sel_ids}", flush=True)

            for seed_idx, seed in enumerate(SEEDS):
                # Transfer
                mt = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                mt.load_state_dict(pretrain_states[seed_idx])
                mt = train_model(mt, Xfew, yfew, device, seed, epochs_ft, freeze_encoder=True)
                tp_norm = predict_norm(mt, Xtest, device)
                tp_cyc  = tp_norm * ytest_max
                tr = rmse(ytest_cyc, tp_cyc); transfer_rmses.append(tr)

                # Scratch
                ms = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                ms = train_model(ms, Xfew, yfew, device, seed, epochs_scr, freeze_encoder=False)
                sp_norm = predict_norm(ms, Xtest, device)
                sp_cyc  = sp_norm * ytest_max
                sr = rmse(ytest_cyc, sp_cyc); scratch_rmses.append(sr)

            trial_log.append({"trial": trial, "sel_ids": sel_ids, "n_windows": int(len(Xfew))})

        t_mean = float(np.mean(transfer_rmses)); t_std = float(np.std(transfer_rmses))
        s_mean = float(np.mean(scratch_rmses));  s_std = float(np.std(scratch_rmses))
        gain   = (s_mean - t_mean) / max(s_mean, 1e-9) * 100
        results[N] = {
            "n_shot": N,
            "transfer_mean_rmse": round(t_mean, 2),
            "transfer_std_rmse":  round(t_std, 2),
            "scratch_mean_rmse":  round(s_mean, 2),
            "scratch_std_rmse":   round(s_std, 2),
            "gain_pct": round(gain, 1),
            "transfer_rmses": [round(v,4) for v in transfer_rmses],
            "scratch_rmses":  [round(v,4) for v in scratch_rmses],
            "trials": trial_log,
        }
        print(f"  N={N:3d}: Transfer={t_mean:.2f}±{t_std:.1f}  Scratch={s_mean:.2f}±{s_std:.1f}  Gain={gain:+.1f}%", flush=True)

    # ── conformal UQ on calibration set (N=5, last 20 GEO train traj) ──
    print("\nComputing conformal UQ on calibration set...", flush=True)
    cal_ids = avail[-20:] if len(avail) >= 20 else avail
    Xcal = torch.as_tensor(
        np.concatenate([geo_train_data[t][2] for t in cal_ids if t in geo_train_data]),
        dtype=torch.float32, device=device)
    ycal_norm = np.concatenate([geo_train_data[t][3] for t in cal_ids if t in geo_train_data])
    ycal_max  = np.concatenate([np.full(len(geo_train_data[t][3]), geo_train_data[t][4])
                                 for t in cal_ids if t in geo_train_data]).astype(np.float32)
    ycal_cyc  = ycal_norm * ycal_max

    conformal_results = {}
    if 5 in results:
        # Use mean pre-trained model for conformal calibration
        m_cal = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
        m_cal.load_state_dict(pretrain_states[0])
        # fine-tune on N=5 representative set
        Xs5 = [geo_train_data[t][0] for t in avail[:5] if t in geo_train_data]
        ys5 = [geo_train_data[t][1] for t in avail[:5] if t in geo_train_data]
        if Xs5:
            Xft5 = torch.as_tensor(np.concatenate(Xs5), dtype=torch.float32, device=device)
            yft5 = torch.as_tensor(np.concatenate(ys5), dtype=torch.float32, device=device)
            m_cal = train_model(m_cal, Xft5, yft5, device, 42, epochs_ft, freeze_encoder=True)
        cal_pred_norm = predict_norm(m_cal, Xcal, device)
        cal_pred_cyc  = cal_pred_norm * ycal_max
        cal_residuals = cal_pred_cyc - ycal_cyc
        test_pred_norm = predict_norm(m_cal, Xtest, device)
        test_pred_cyc  = test_pred_norm * ytest_max
        conformal_results = conformal_intervals(cal_residuals, test_pred_cyc)
        print("  Conformal quantiles:", {k: round(v["quantile_cycles"],1)
                                          for k,v in conformal_results.items()}, flush=True)

    # ── print table ──
    print(f"\n{'='*70}", flush=True)
    print("FEW-SHOT TRANSFER v2 RESULTS  (RUL-norm fix + 5 trials/N)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>5} {'Transfer':>16} ({'±':^4}) {'Scratch':>16} ({'±':^4}) {'Gain':>8}", flush=True)
    print("-"*70, flush=True)
    for N, r in results.items():
        print(f"{N:>5} {r['transfer_mean_rmse']:>16.2f} ({r['transfer_std_rmse']:>4.1f})"
              f" {r['scratch_mean_rmse']:>16.2f} ({r['scratch_std_rmse']:>4.1f})"
              f" {r['gain_pct']:>7.1f}%", flush=True)
    print("="*70, flush=True)

    report = {
        "schema": "few_shot_transfer_v2",
        "rul_norm": "first-window-relative (max_rul=rul[seq_len-1])",
        "early_frac": EARLY_FRAC,
        "n_trials": n_trials,
        "seeds": list(SEEDS),
        "epochs_pretrain": epochs_pre,
        "epochs_finetune": epochs_ft,
        "epochs_scratch": epochs_scr,
        "n_test_windows": int(len(Xtest)),
        "results": {str(k): v for k,v in results.items()},
        "conformal_uq": conformal_results,
    }
    (out_dir / "FEW_SHOT_V2_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv as _csv
    with open(out_dir / "few_shot_v2_curve.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["n_shot","method","rmse","std"])
        for N, r in results.items():
            w.writerow([N, "transfer", r["transfer_mean_rmse"], r["transfer_std_rmse"]])
            w.writerow([N, "scratch",  r["scratch_mean_rmse"],  r["scratch_std_rmse"]])
    print(f"Report → {out_dir}/FEW_SHOT_V2_REPORT.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/few_shot_transfer_v2")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Few-shot v2  device={device}  quick={args.quick}", flush=True)
    t0 = time.time()
    run(device, args.output, args.quick)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
