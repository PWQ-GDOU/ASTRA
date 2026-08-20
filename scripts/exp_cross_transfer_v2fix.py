"""Cross-component transfer: GEO battery / nozzle → FEMTO bearing (reaction-wheel proxy).

This script demonstrates multi-direction transfer learning:
  Source A: GEO satellite PINN battery (60 traj, SOH degradation, 0-1 RUL)
  Source B: nozzle ablation (COMSOL, time-to-failure, normalized 0-1)
  Target  : FEMTO bearing (6 bearings LOO, vibration RUL proxy)

Shared representation: normalized degradation curve shape (monotone 0→1 RUL).
The encoder learns the universal "remaining life decreases monotonically" prior.

Protocol:
  - Outer LOO on FEMTO bearings (same as RW v1/v2)
  - Pre-train GRU encoder on source domain (GEO battery or nozzle)
  - Fine-tune head+encoder on N=1,2,3 bearing early-life windows (EARLY_FRAC=0.20)
  - Compare: Transfer vs Scratch on held-out test bearing
"""
from __future__ import annotations

import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.femto_strict import (
    FEATURE_GROUPS, FemtoWindowSet, fit_scaler, load_femto_zip, make_windows,
)

# ── constants ────────────────────────────────────────────────────────────
SEEDS       = (42, 123, 456, 2026, 3407)
EARLY_FRAC  = 0.20
SEQ_LEN     = 20
HIDDEN      = 64
DROPOUT     = 0.1
EPOCHS_PRE  = 100
EPOCHS_FT   = 100
EPOCHS_SCR  = 200
N_SHOTS     = (1, 2, 3)
N_TRIALS    = 2
COMMON_EP   = 19
GEO_CSV     = "data/processed/geo_battery_60traj/geo_battery_60traj.csv"
NOZZLE_MT_CSV = "data/processed/nozzle_mt_200/nozzle_sim_200traj.csv"

# ── shared GRU encoder ───────────────────────────────────────────────────
class DegradationEncoder(nn.Module):
    """1-feature GRU encoder for normalized degradation sequences."""
    def __init__(self, n_in=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.hidden = hidden

    def forward(self, x):
        h, _ = self.encoder(x)
        return h[:, -1]   # (B, hidden)


class TransferRULModel(nn.Module):
    def __init__(self, n_features=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.encoder = DegradationEncoder(n_features, hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, hidden),   # +1 for cycle_pos
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, cycle_pos=None):
        z = self.encoder(x)
        if cycle_pos is None:
            cycle_pos = torch.zeros(z.size(0), 1, device=z.device)
        return self.head(torch.cat([z, cycle_pos], dim=-1)).squeeze(-1)


# ── utilities ────────────────────────────────────────────────────────────
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rmse(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))

def train_epoch(model, X, y, cp, opt, scheduler, rng, batch=256):
    model.train()
    idx = rng.permutation(len(X))
    for s in range(0, len(idx), batch):
        b = idx[s:s+batch]
        out = model(X[b], cp[b])
        loss = F.smooth_l1_loss(out, y[b], beta=0.1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step()

def train_model(model, X, y, cp, seed, epochs, freeze_enc=False, lr=3e-4):
    seed_all(seed)
    if freeze_enc:
        for p in model.encoder.parameters(): p.requires_grad_(False)
    else:
        for p in model.parameters(): p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        train_epoch(model, X, y, cp, opt, sched, rng)
    for p in model.parameters(): p.requires_grad_(True)
    return model

def predict(model, X, cp, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X, cp).cpu().numpy()

# ── source domain data loaders ───────────────────────────────────────────

def load_geo_battery_source(geo_csv, seq_len=SEQ_LEN, eol_soh=0.80, device="cpu"):
    """Load GEO PINN battery trajectories as source pre-training data."""
    import pandas as pd
    df = pd.read_csv(geo_csv)
    Xs, ys, cps = [], [], []
    for tid, grp in df.groupby("trajectory_id"):
        soh = grp["soh"].values.astype(np.float32)
        cycle = grp["cycle"].values.astype(np.float32)
        hits = np.flatnonzero(soh <= eol_soh)
        if hits.size == 0: continue
        eol = float(cycle[hits[0]])
        rul = np.maximum(eol - cycle, 0.0)
        first_w = seq_len - 1
        max_rul = float(rul[first_w]) if rul[first_w] > 0 else float(rul[0])
        if max_rul <= 0: continue
        rul_norm = rul / max_rul
        for end in range(seq_len - 1, len(soh)):
            Xs.append(soh[end-seq_len+1:end+1])
            ys.append(rul_norm[end])
            cps.append(float(cycle[end]) / max(eol, 1.0))
    X = torch.as_tensor(np.array(Xs, np.float32)[:,:,None], dtype=torch.float32, device=device)
    y = torch.as_tensor(np.array(ys, np.float32), dtype=torch.float32, device=device)
    cp = torch.as_tensor(np.array(cps, np.float32)[:,None], dtype=torch.float32, device=device)
    if len(X) > 3000:
        idx = torch.randperm(len(X))[:3000]
        X, y, cp = X[idx], y[idx], cp[idx]
    print(f"  GEO battery source: {len(X)} windows  RUL_norm {y.min():.3f}-{y.max():.3f}", flush=True)
    return X, y, cp


def load_nozzle_source(seq_len=SEQ_LEN, device="cpu"):
    """Load nozzle ablation trajectories (COMSOL, 200-trajectory strict multi-
    trajectory set) as source pre-training data (normalized depth proxy).

    Uses the same strict loader as the nozzle SOTA validation pipeline
    (src/data/nozzle_multitrajectory.py) so column names, right-censoring,
    and threshold-crossing interpolation are handled consistently instead
    of matching English substrings against the frozen single-trajectory
    Chinese-column CSV (data/raw/nozzle_ablation_full.csv), which this
    function used to read and always failed to parse.
    """
    nozzle_path = Path(NOZZLE_MT_CSV)
    if not nozzle_path.exists():
        print("  Nozzle multi-trajectory data not found, skipping", flush=True)
        return None, None, None
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.nozzle_multitrajectory import load_nozzle_trajectories
    trajs = load_nozzle_trajectories(nozzle_path, feature_tier="estimated")
    Xs, ys, cps = [], [], []
    for traj in trajs.values():
        if not traj.target_mask.all():
            continue  # right-censored: no defined RUL, skip for source pretraining
        depth = traj.depth_mm.astype(np.float32)
        rul = traj.rul_s.astype(np.float32)
        first_w = seq_len - 1
        if len(depth) <= first_w:
            continue
        max_rul = float(rul[first_w]) if rul[first_w] > 0 else float(rul[0])
        if max_rul <= 0:
            continue
        rul_norm = rul / max_rul
        # depth rises with damage; health falls with damage (matches soh/rms convention)
        dep_norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-9)
        health_norm = 1.0 - dep_norm
        max_t = float(traj.time_s[-1])
        for end in range(first_w, len(health_norm)):
            Xs.append(health_norm[end-seq_len+1:end+1])
            ys.append(rul_norm[end])
            cps.append(traj.time_s[end] / max_t)
    if not Xs:
        return None, None, None
    X = torch.as_tensor(np.array(Xs, np.float32)[:,:,None], dtype=torch.float32, device=device)
    y = torch.as_tensor(np.array(ys, np.float32), dtype=torch.float32, device=device)
    cp = torch.as_tensor(np.array(cps, np.float32)[:,None], dtype=torch.float32, device=device)
    if len(X) > 3000:
        idx = torch.randperm(len(X))[:3000]
        X, y, cp = X[idx], y[idx], cp[idx]
    print(f"  Nozzle source: {len(X)} windows from {len(trajs)} trajectories  RUL_norm {y.min():.3f}-{y.max():.3f}", flush=True)
    return X, y, cp


# ── FEMTO bearing data ───────────────────────────────────────────────────

def load_femto_early(series_list, seq_len=SEQ_LEN, feature_group="base",
                     common_ep=COMMON_EP, early_frac=EARLY_FRAC, rul_scale=None):
    """Return early-life (SOH-like normalized RMS) and full-life windows for each bearing."""
    bearing_data = {}
    if rul_scale is None:
        rul_scale = max(max(float(s.raw_rul[0]) for s in series_list), 1.0)
    scaler = fit_scaler(series_list, tuple(FEATURE_GROUPS[feature_group]))
    for s in series_list:
        ws = make_windows([s], seq_len, scaler, min_endpoint=common_ep, rul_scale=rul_scale)
        if len(ws) == 0: continue
        n_early = max(seq_len + 1, int(len(ws) * early_frac))
        # early-life subset
        early = ws.subset(np.arange(n_early))
        full  = ws
        # use acc_x_rms trend as "SOH proxy" — first feature, normalize 0-1
        rms = early.X[:, :, 0]  # (N_early, seq_len)
        rms_min, rms_max = rms.min(), rms.max()
        rms_norm = (rms - rms_min) / max(rms_max - rms_min, 1e-9)
        health_norm = 1.0 - rms_norm  # rms rises with damage; health falls with damage (matches soh convention)
        # normalized RUL
        rul_norm_e = early.rul / rul_scale
        rul_norm_f = full.rul / rul_scale
        cp_e = early.endpoints.astype(np.float32) / rul_scale
        cp_f = full.endpoints.astype(np.float32) / rul_scale
        # rms-norm for full windows (1 feature, compatible with encoder)
        rms_full = full.X[:, :, 0]
        r_min, r_max = rms_full.min(), rms_full.max()
        rms_full_norm = (rms_full - r_min) / max(r_max - r_min, 1e-9)
        health_full_norm = 1.0 - rms_full_norm
        bearing_data[s.name] = {
            "X_early": health_norm[:, :, None].astype(np.float32),  # (N_e, L, 1) health=1-rms
            "y_early": rul_norm_e.astype(np.float32),
            "cp_early": cp_e[:, None],
            "X_full_1feat": health_full_norm[:, :, None].astype(np.float32),  # (N_f, L, 1) health=1-rms
            "X_full":  full.X.astype(np.float32),
            "y_full":  rul_norm_f.astype(np.float32),
            "cp_full": cp_f[:, None],
            "rul_scale": rul_scale,
            "n_total": len(full),
        }
    return bearing_data, rul_scale


# ── LOO experiment ───────────────────────────────────────────────────────

def run_cross_transfer_loo(source_X, source_y, source_cp, bearing_data,
                           rul_scale, source_name, device, n_features_target=1):
    """LOO over FEMTO bearings, each time transferring from source domain."""
    bearing_names = list(bearing_data.keys())
    results = {}

    print(f"\nSource: {source_name}  ({len(source_X)} windows)", flush=True)

    # pre-train on source domain once per seed
    pretrain_states = []
    for seed in SEEDS:
        m = TransferRULModel(n_features=1, hidden=HIDDEN, dropout=DROPOUT).to(device)
        m = train_model(m, source_X, source_y, source_cp, seed, EPOCHS_PRE)
        pretrain_states.append({k: v.cpu().clone() for k, v in m.state_dict().items()})
    print(f"  Pre-training done ({EPOCHS_PRE} epochs × {len(SEEDS)} seeds)", flush=True)

    for holdout in bearing_names:
        train_bearings = [b for b in bearing_names if b != holdout]
        print(f"\n  holdout={holdout}", flush=True)

        # avail training bearings for few-shot
        avail = [b for b in train_bearings if b in bearing_data]
        test_bd = bearing_data[holdout]
        Xt = torch.as_tensor(test_bd["X_early"], dtype=torch.float32, device=device)
        yt = test_bd["y_full"] * rul_scale  # de-normalized
        cpt = torch.as_tensor(test_bd["cp_full"], dtype=torch.float32, device=device)
        Xtest = torch.as_tensor(test_bd["X_early"][:len(test_bd["y_full"])], dtype=torch.float32, device=device)
        # use full X (1 feature) for test
        n_full = test_bd["n_total"]
        X_full_1feat = torch.as_tensor(test_bd["X_early"] if len(test_bd["X_early"]) >= n_full
                                        else test_bd["X_full"][:,:,0:1], dtype=torch.float32, device=device)

        fold_results = {}
        for N in N_SHOTS:
            if N > len(avail):
                continue
            tr_rmses, sc_rmses = [], []
            for trial in range(N_TRIALS):
                rng_t = np.random.default_rng(trial * 999 + N + hash(holdout) % 10000)
                sel = [avail[i] for i in rng_t.choice(len(avail), N, replace=False)]
                Xf = np.concatenate([bearing_data[b]["X_early"] for b in sel])
                yf = np.concatenate([bearing_data[b]["y_early"] for b in sel])
                cf = np.concatenate([bearing_data[b]["cp_early"] for b in sel])
                Xf_t = torch.as_tensor(Xf, dtype=torch.float32, device=device)
                yf_t = torch.as_tensor(yf, dtype=torch.float32, device=device)
                cf_t = torch.as_tensor(cf, dtype=torch.float32, device=device)
                # build full test set
                full_rul = bearing_data[holdout]["y_full"] * rul_scale
                full_cp  = torch.as_tensor(bearing_data[holdout]["cp_full"],
                                           dtype=torch.float32, device=device)
                full_X   = torch.as_tensor(
                    bearing_data[holdout]["X_full_1feat"],
                    dtype=torch.float32, device=device)

                for seed_i, seed in enumerate(SEEDS):
                    # Transfer
                    mt = TransferRULModel(n_features=1, hidden=HIDDEN, dropout=DROPOUT).to(device)
                    mt.load_state_dict(pretrain_states[seed_i])
                    mt = train_model(mt, Xf_t, yf_t, cf_t, seed, EPOCHS_FT, freeze_enc=True)
                    tp = predict(mt, full_X, full_cp, device) * rul_scale
                    tr_rmses.append(rmse(full_rul, tp))

                    # Scratch
                    ms = TransferRULModel(n_features=1, hidden=HIDDEN, dropout=DROPOUT).to(device)
                    ms = train_model(ms, Xf_t, yf_t, cf_t, seed, EPOCHS_SCR, freeze_enc=False)
                    sp = predict(ms, full_X, full_cp, device) * rul_scale
                    sc_rmses.append(rmse(full_rul, sp))

            t_mean = float(np.mean(tr_rmses)); s_mean = float(np.mean(sc_rmses))
            gain = (s_mean - t_mean) / max(s_mean, 1e-9) * 100
            fold_results[N] = {"transfer_rmse": round(t_mean,1),
                               "scratch_rmse": round(s_mean,1),
                               "gain_pct": round(gain,1)}
            print(f"    N={N}: Transfer={t_mean:.1f}  Scratch={s_mean:.1f}  Gain={gain:+.1f}%", flush=True)
        results[holdout] = fold_results

    return results


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="outputs/cross_component_transfer")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--data-zip", default="data/processed/femto_bearing.zip")
    args = parser.parse_args()

    device = args.device if (torch.cuda.is_available() and args.device != "cpu") else "cpu"
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cross-component transfer  device={device}", flush=True)
    t0 = time.time()

    # ── load FEMTO ──
    print("\nLoading FEMTO bearing data...", flush=True)
    series_dict = load_femto_zip(args.data_zip, feature_group="base")
    all_series = list(series_dict.values())
    rul_scale = max(max(float(s.raw_rul[0]) for s in all_series), 1.0)
    bearing_data, rul_scale = load_femto_early(all_series, SEQ_LEN, "base",
                                               COMMON_EP, EARLY_FRAC, rul_scale)
    print(f"Loaded {len(bearing_data)} bearings  rul_scale={rul_scale:.0f}", flush=True)

    all_results = {}

    # ── Source A: GEO battery ──
    print("\nLoading GEO battery source...", flush=True)
    if Path(GEO_CSV).exists():
        Xg, yg, cpg = load_geo_battery_source(GEO_CSV, SEQ_LEN, device=device)
        res_geo = run_cross_transfer_loo(Xg, yg, cpg, bearing_data, rul_scale,
                                         "GEO_battery", device)
        all_results["GEO_battery"] = res_geo
    else:
        print("  GEO CSV not found, skipping", flush=True)

    # ── Source B: Nozzle ──
    print("\nLoading nozzle source...", flush=True)
    Xn, yn, cpn = load_nozzle_source(SEQ_LEN, device=device)
    if Xn is not None:
        res_nozzle = run_cross_transfer_loo(Xn, yn, cpn, bearing_data, rul_scale,
                                            "nozzle", device)
        all_results["nozzle"] = res_nozzle

    # ── aggregate and print ──
    print(f"\n{'='*70}", flush=True)
    print("CROSS-COMPONENT TRANSFER — macro LOO results", flush=True)
    print(f"{'='*70}", flush=True)
    for src_name, src_res in all_results.items():
        print(f"\nSource: {src_name}", flush=True)
        for N in N_SHOTS:
            tr_list = [src_res[b][N]["transfer_rmse"] for b in src_res if N in src_res[b]]
            sc_list = [src_res[b][N]["scratch_rmse"]  for b in src_res if N in src_res[b]]
            if not tr_list: continue
            t_mac = float(np.mean(tr_list)); s_mac = float(np.mean(sc_list))
            gain = (s_mac - t_mac) / max(s_mac, 1e-9) * 100
            print(f"  N={N:2d}: Transfer={t_mac:.1f}  Scratch={s_mac:.1f}  Gain={gain:+.1f}%", flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)

    report = {
        "schema": "cross_component_transfer_v1",
        "sources": list(all_results.keys()),
        "target": "FEMTO bearing (reaction-wheel proxy)",
        "early_frac": EARLY_FRAC,
        "n_shots": list(N_SHOTS),
        "n_trials": N_TRIALS,
        "seeds": list(SEEDS),
        "results": all_results,
    }
    (out_dir / "CROSS_TRANSFER_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report → {out_dir}/CROSS_TRANSFER_REPORT.json", flush=True)


if __name__ == "__main__":
    main()
