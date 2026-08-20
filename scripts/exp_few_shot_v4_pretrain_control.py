"""Few-shot v4: no-pretraining isolation control.

Phase1 v3 has two arms: transfer (pretrained encoder, FROZEN) and scratch
(random init, unfrozen). That comparison conflates two effects:
  1. Does the pretrained encoder's representation help at all?
  2. Does freezing the encoder (vs fine-tuning it) matter?

v4 adds a third arm to separate them:
  - transfer_frozen    : pretrained init, encoder FROZEN     (= v3 "transfer")
  - transfer_unfrozen  : pretrained init, encoder fine-tuned (NEW control)
  - scratch             : random init,     encoder fine-tuned (= v3 "scratch")

Reading:
  transfer_frozen   vs scratch            -> pretraining benefit (v3's headline number)
  transfer_frozen   vs transfer_unfrozen  -> does freezing itself matter, given pretraining?
  transfer_unfrozen vs scratch            -> pretraining benefit when both are allowed
                                              to adapt the encoder (isolates init-only effect)

All other methodology (splits, early stopping, seeds, trials, Ridge baseline,
conformal UQ) is unchanged from v3.
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

# ── constants (identical to v3) ──────────────────────────────────────────
EARLY_FRAC   = 0.20
EPOCHS_PRE   = 400
EPOCHS_FT    = 300
EPOCHS_SCR   = 500
SEQ_LEN      = 16
HIDDEN       = 64
DROPOUT      = 0.1
N_SHOTS      = (1, 2, 3, 5, 10, 20)
N_TRIALS     = 5
N_TEST       = 10
SEEDS        = (42, 123, 456, 2026, 3407)
NASA_AUX     = ("B0042","B0043","B0044","B0046","B0047","B0048")
GEO_CSV      = "data/processed/geo_battery_60traj/geo_battery_60traj.csv"
NASA_DIR     = "data/processed/nasa_battery/5. Battery Data Set"
VAL_FRAC     = 0.20
MIN_VAL      = 4
PATIENCE     = 10
EVAL_EVERY   = 5
RIDGE_ALPHA  = 1.0

# ── model (identical to v3) ──────────────────────────────────────────────
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

# ── data (identical to v3) ────────────────────────────────────────────────
def _rul_norm_from_first_window(rul: np.ndarray, seq_len: int) -> tuple[np.ndarray, float]:
    first_window_idx = seq_len - 1
    max_rul = float(rul[first_window_idx]) if len(rul) > first_window_idx and rul[first_window_idx] > 0 else float(rul[0])
    if max_rul <= 0:
        max_rul = float(np.nanmax(rul[rul > 0])) if np.any(rul > 0) else 1.0
    return rul / max_rul, max_rul

def soh_from_geo(df_traj, seq_len=SEQ_LEN, eol_soh=0.80, early_frac=EARLY_FRAC):
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
    n_early = max(seq_len + 1, int(len(soh) * early_frac))
    windows_e, targets_e = [], []
    for end in range(seq_len-1, n_early):
        windows_e.append(soh[end-seq_len+1:end+1])
        targets_e.append(rul_norm[end])
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

def train_model_fixed(model, X, y, device, seed, epochs, batch=512, lr=3e-4):
    """v3 fixed-epoch trainer -- used only for NASA pre-training."""
    seed_all(seed)
    for p in model.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
    return model

def train_model_es(model, Xtr, ytr, Xval, yval, device, seed, max_epochs,
                    freeze_encoder=False, batch=512, lr=3e-4,
                    patience=PATIENCE, eval_every=EVAL_EVERY):
    """v3 few-shot trainer with validation-based early stopping (unchanged)."""
    seed_all(seed)
    if freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad_(False)
    else:
        for p in model.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs))
    rng = np.random.default_rng(seed)
    has_val = Xval is not None and len(Xval) > 0
    best_val = float("inf"); best_state = None; best_epoch = 0; no_improve = 0
    stopped_epoch = max_epochs - 1
    for epoch in range(max_epochs):
        model.train()
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(idx), batch):
            b = idx[s:s+batch]
            out = model(Xtr[b])
            loss = F.smooth_l1_loss(out, ytr[b], beta=0.1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if has_val and (epoch % eval_every == 0 or epoch == max_epochs - 1):
            model.eval()
            with torch.no_grad():
                vp = model(Xval)
                vloss = float(torch.sqrt(torch.mean((vp - yval) ** 2)))
            model.train()
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch; no_improve = 0
            else:
                no_improve += eval_every
            if no_improve >= patience:
                stopped_epoch = epoch
                break
    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    for p in model.parameters(): p.requires_grad_(True)
    diag = {
        "has_val": has_val,
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "best_val_rmse_norm": round(best_val, 4) if has_val else None,
    }
    return model, diag

def predict_norm(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy()

def make_val_split(sel_ids, geo_train_data, seed):
    """Identical to v3: N>=2 trajectory holdout, N==1 chronological gapped split."""
    if len(sel_ids) >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sel_ids))
        n_val_traj = max(1, int(round(len(sel_ids) * VAL_FRAC)))
        n_val_traj = min(n_val_traj, len(sel_ids) - 1)
        val_ids = [sel_ids[i] for i in idx[:n_val_traj]]
        tr_ids  = [sel_ids[i] for i in idx[n_val_traj:]]
        Xtr = np.concatenate([geo_train_data[t][0] for t in tr_ids])
        ytr = np.concatenate([geo_train_data[t][1] for t in tr_ids])
        Xval = np.concatenate([geo_train_data[t][0] for t in val_ids])
        yval = np.concatenate([geo_train_data[t][1] for t in val_ids])
        return Xtr, ytr, Xval, yval, "trajectory_holdout"
    else:
        Xe, ye = geo_train_data[sel_ids[0]][0], geo_train_data[sel_ids[0]][1]
        n = len(Xe)
        n_val = max(MIN_VAL, int(round(n * VAL_FRAC)))
        gap = SEQ_LEN
        if n - n_val - gap >= MIN_VAL:
            Xtr, ytr = Xe[:n - n_val - gap], ye[:n - n_val - gap]
            Xval, yval = Xe[n - n_val:], ye[n - n_val:]
            return Xtr, ytr, Xval, yval, "chronological_holdout_gapped"
        else:
            return Xe, ye, None, None, "no_val_fixed_epochs"

# ── classical ridge baseline (identical to v3) ────────────────────────────
def ridge_features(X: np.ndarray) -> np.ndarray:
    x = X[:, :, 0].astype(np.float64)
    mean = x.mean(axis=1)
    last = x[:, -1]
    std = x.std(axis=1)
    t = np.arange(x.shape[1], dtype=np.float64)
    t_c = t - t.mean()
    denom = float((t_c ** 2).sum())
    slope = (x * t_c).sum(axis=1) / denom
    return np.stack([mean, last, std, slope], axis=1)

def fit_ridge(X: np.ndarray, y: np.ndarray, alpha=RIDGE_ALPHA):
    Feat = ridge_features(X)
    Fb = np.c_[np.ones(len(Feat)), Feat]
    I = np.eye(Fb.shape[1]); I[0, 0] = 0.0
    return np.linalg.solve(Fb.T @ Fb + alpha * I, Fb.T @ y)

def predict_ridge(X: np.ndarray, beta) -> np.ndarray:
    Feat = ridge_features(X)
    Fb = np.c_[np.ones(len(Feat)), Feat]
    return np.clip(Fb @ beta, 0.0, None).astype(np.float32)


# ── main experiment ───────────────────────────────────────────────────────
def run(device, out_dir, quick=False):
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            geo_train_data[tid] = d

    geo_test_Xf, geo_test_yf, geo_test_maxruls = [], [], []
    for tid in test_ids:
        sub = geo_df[geo_df["trajectory_id"]==tid]
        d = soh_from_geo(sub, SEQ_LEN)
        if d is not None:
            geo_test_Xf.append(d[2]); geo_test_yf.append(d[3])
            geo_test_maxruls.append(np.full(len(d[3]), d[4], dtype=np.float32))

    Xtest_np = np.concatenate(geo_test_Xf)
    Xtest = torch.as_tensor(Xtest_np, dtype=torch.float32, device=device)
    ytest_norm  = np.concatenate(geo_test_yf)
    ytest_max   = np.concatenate(geo_test_maxruls)
    ytest_cyc   = ytest_norm * ytest_max
    print(f"GEO test: {len(Xtest)} windows, RUL range {ytest_cyc.min():.0f}-{ytest_cyc.max():.0f} cycles", flush=True)

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

    epochs_pre = EPOCHS_PRE if not quick else 50
    epochs_ft  = EPOCHS_FT  if not quick else 50
    epochs_scr = EPOCHS_SCR if not quick else 100
    print(f"\nPre-training on NASA ({len(NASA_AUX)} cells, {epochs_pre} epochs)...", flush=True)
    pretrain_states = []
    for seed in SEEDS:
        m = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
        m = train_model_fixed(m, Xnasa, ynasa, device, seed, epochs_pre)
        pretrain_states.append({k: v.cpu().clone() for k,v in m.state_dict().items()})
    print("Pre-training done.", flush=True)

    avail = [t for t in train_ids if t in geo_train_data]
    results = {}
    n_trials = N_TRIALS if not quick else 2

    for N in N_SHOTS:
        if N > len(avail):
            print(f"  N={N}: not enough trajectories ({len(avail)} avail)", flush=True)
            continue
        transfer_frozen_rmses, transfer_unfrozen_rmses, scratch_rmses, ridge_rmses = [], [], [], []
        trial_log = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(trial * 7777 + N)
            chosen = rng_t.choice(len(avail), N, replace=False)
            sel_ids = [avail[i] for i in chosen]
            split_seed = trial * 9001 + N
            Xtr_np, ytr_np, Xval_np, yval_np, split_kind = make_val_split(sel_ids, geo_train_data, split_seed)
            print(f"  N={N} trial={trial}: sel={sel_ids} split={split_kind} "
                  f"n_train={len(Xtr_np)} n_val={0 if Xval_np is None else len(Xval_np)}", flush=True)

            Xtr = torch.as_tensor(Xtr_np, dtype=torch.float32, device=device)
            ytr = torch.as_tensor(ytr_np, dtype=torch.float32, device=device)
            Xval = torch.as_tensor(Xval_np, dtype=torch.float32, device=device) if Xval_np is not None else None
            yval = torch.as_tensor(yval_np, dtype=torch.float32, device=device) if yval_np is not None else None

            Xfull_np = np.concatenate([geo_train_data[t][0] for t in sel_ids])
            yfull_np = np.concatenate([geo_train_data[t][1] for t in sel_ids])
            beta = fit_ridge(Xfull_np, yfull_np)
            rp_norm = predict_ridge(Xtest_np, beta)
            rp_cyc = rp_norm * ytest_max
            rr = rmse(ytest_cyc, rp_cyc); ridge_rmses.append(rr)

            trial_diag = {"trial": trial, "sel_ids": sel_ids, "split_kind": split_kind,
                           "n_train": int(len(Xtr_np)),
                           "n_val": 0 if Xval_np is None else int(len(Xval_np)),
                           "ridge_rmse": round(rr, 4), "seeds": {}}

            for seed_idx, seed in enumerate(SEEDS):
                # arm 1: transfer_frozen (= v3 "transfer") — pretrained encoder, frozen
                mtf = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                mtf.load_state_dict(pretrain_states[seed_idx])
                mtf, diag_tf = train_model_es(mtf, Xtr, ytr, Xval, yval, device, seed, epochs_ft,
                                               freeze_encoder=True)
                tf_norm = predict_norm(mtf, Xtest, device)
                tf_cyc  = tf_norm * ytest_max
                tfr = rmse(ytest_cyc, tf_cyc); transfer_frozen_rmses.append(tfr)

                # arm 2: transfer_unfrozen — pretrained encoder, but fully fine-tuned
                #   (isolates: does freezing matter, given the same pretrained init?)
                mtu = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                mtu.load_state_dict(pretrain_states[seed_idx])
                mtu, diag_tu = train_model_es(mtu, Xtr, ytr, Xval, yval, device, seed, epochs_ft,
                                               freeze_encoder=False)
                tu_norm = predict_norm(mtu, Xtest, device)
                tu_cyc  = tu_norm * ytest_max
                tur = rmse(ytest_cyc, tu_cyc); transfer_unfrozen_rmses.append(tur)

                # arm 3: scratch (= v3 "scratch") — random init, fully trained
                ms = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                ms, diag_s = train_model_es(ms, Xtr, ytr, Xval, yval, device, seed, epochs_scr,
                                             freeze_encoder=False)
                sp_norm = predict_norm(ms, Xtest, device)
                sp_cyc  = sp_norm * ytest_max
                sr = rmse(ytest_cyc, sp_cyc); scratch_rmses.append(sr)

                trial_diag["seeds"][str(seed)] = {
                    "transfer_frozen_rmse": round(tfr, 4), "transfer_frozen_diag": diag_tf,
                    "transfer_unfrozen_rmse": round(tur, 4), "transfer_unfrozen_diag": diag_tu,
                    "scratch_rmse": round(sr, 4), "scratch_diag": diag_s,
                }
            trial_log.append(trial_diag)

        tf_mean = float(np.mean(transfer_frozen_rmses)); tf_std = float(np.std(transfer_frozen_rmses))
        tu_mean = float(np.mean(transfer_unfrozen_rmses)); tu_std = float(np.std(transfer_unfrozen_rmses))
        s_mean = float(np.mean(scratch_rmses));  s_std = float(np.std(scratch_rmses))
        r_mean = float(np.mean(ridge_rmses));    r_std = float(np.std(ridge_rmses))
        # pretraining_benefit_pct: gain of frozen-pretrained transfer over scratch (same as v3 metric)
        pretrain_benefit_pct = (s_mean - tf_mean) / max(s_mean, 1e-9) * 100
        # freezing_cost_pct: how much frozen loses/gains vs fully fine-tuning the SAME pretrained init
        #   positive = freezing helps, negative = freezing hurts (unfreezing would have been better)
        freezing_effect_pct = (tu_mean - tf_mean) / max(tu_mean, 1e-9) * 100
        results[N] = {
            "n_shot": N,
            "transfer_frozen_mean_rmse": round(tf_mean, 2), "transfer_frozen_std_rmse": round(tf_std, 2),
            "transfer_unfrozen_mean_rmse": round(tu_mean, 2), "transfer_unfrozen_std_rmse": round(tu_std, 2),
            "scratch_mean_rmse":  round(s_mean, 2), "scratch_std_rmse":  round(s_std, 2),
            "ridge_mean_rmse":    round(r_mean, 2), "ridge_std_rmse":    round(r_std, 2),
            "pretrain_benefit_pct_vs_scratch": round(pretrain_benefit_pct, 1),
            "freezing_effect_pct_vs_unfrozen_transfer": round(freezing_effect_pct, 1),
            "transfer_frozen_rmses": [round(v,4) for v in transfer_frozen_rmses],
            "transfer_unfrozen_rmses": [round(v,4) for v in transfer_unfrozen_rmses],
            "scratch_rmses":  [round(v,4) for v in scratch_rmses],
            "ridge_rmses":    [round(v,4) for v in ridge_rmses],
            "trials": trial_log,
        }
        print(f"  N={N:3d}: TransferFrozen={tf_mean:.2f}±{tf_std:.1f}  TransferUnfrozen={tu_mean:.2f}±{tu_std:.1f}"
              f"  Scratch={s_mean:.2f}±{s_std:.1f}  Ridge={r_mean:.2f}±{r_std:.1f}"
              f"  PretrainBenefit={pretrain_benefit_pct:+.1f}%  FreezingEffect={freezing_effect_pct:+.1f}%",
              flush=True)

    print(f"\n{'='*100}", flush=True)
    print("FEW-SHOT v4 PRETRAINING-CONTROL RESULTS", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'N':>5} {'TransFrozen':>14} {'TransUnfroz':>14} {'Scratch':>14} {'Ridge':>14}"
          f" {'PretrainBenefit':>16} {'FreezeEffect':>14}", flush=True)
    print("-"*100, flush=True)
    for N, r in results.items():
        print(f"{N:>5} {r['transfer_frozen_mean_rmse']:>9.2f}±{r['transfer_frozen_std_rmse']:<4.1f}"
              f" {r['transfer_unfrozen_mean_rmse']:>9.2f}±{r['transfer_unfrozen_std_rmse']:<4.1f}"
              f" {r['scratch_mean_rmse']:>9.2f}±{r['scratch_std_rmse']:<4.1f}"
              f" {r['ridge_mean_rmse']:>9.2f}±{r['ridge_std_rmse']:<4.1f}"
              f" {r['pretrain_benefit_pct_vs_scratch']:>15.1f}% {r['freezing_effect_pct_vs_unfrozen_transfer']:>13.1f}%",
              flush=True)
    print("="*100, flush=True)

    report = {
        "schema": "few_shot_v4_pretrain_control",
        "purpose": ("Isolates transfer-pretraining benefit from encoder-freezing structure. "
                    "transfer_frozen = v3 'transfer' (pretrained+frozen). "
                    "transfer_unfrozen = pretrained init, but fully fine-tuned (no freeze). "
                    "scratch = random init, fully trained (v3 'scratch', no-pretraining control)."),
        "rul_norm": "first-window-relative (max_rul=rul[seq_len-1])",
        "early_stopping": {"val_frac": VAL_FRAC, "min_val": MIN_VAL, "patience": PATIENCE, "eval_every": EVAL_EVERY},
        "early_frac": EARLY_FRAC,
        "n_trials": n_trials,
        "seeds": list(SEEDS),
        "epochs_pretrain": epochs_pre,
        "epochs_finetune_max": epochs_ft,
        "epochs_scratch_max": epochs_scr,
        "ridge_alpha": RIDGE_ALPHA,
        "n_test_windows": int(len(Xtest)),
        "results": {str(k): v for k,v in results.items()},
    }
    (out_dir / "FEW_SHOT_V4_PRETRAIN_CONTROL_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv as _csv
    with open(out_dir / "few_shot_v4_pretrain_control_curve.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["n_shot","method","rmse","std"])
        for N, r in results.items():
            w.writerow([N, "transfer_frozen",   r["transfer_frozen_mean_rmse"],   r["transfer_frozen_std_rmse"]])
            w.writerow([N, "transfer_unfrozen",  r["transfer_unfrozen_mean_rmse"], r["transfer_unfrozen_std_rmse"]])
            w.writerow([N, "scratch",            r["scratch_mean_rmse"],           r["scratch_std_rmse"]])
            w.writerow([N, "ridge",              r["ridge_mean_rmse"],             r["ridge_std_rmse"]])
    print(f"Report → {out_dir}/FEW_SHOT_V4_PRETRAIN_CONTROL_REPORT.json", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/few_shot_v4_pretrain_control")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Few-shot v4 pretrain-control  device={device}  quick={args.quick}", flush=True)
    t0 = time.time()
    run(device, args.output, args.quick)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
