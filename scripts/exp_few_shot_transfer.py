"""Few-shot transfer learning: NASA battery (public) → GEO satellite battery (spacecraft sim).

TRUE FEW-SHOT SCENARIO
-----------------------
Reality: a satellite is launched; we only observe the FIRST K cycles of its battery.
Question: can we predict its full remaining life (RUL) accurately?

Source domain  : NASA auxiliary batteries (B0042-B0048, 6 event-observed cells)
                 Pre-train encoder to learn degradation curve shapes.
Target domain  : GEO satellite PINN simulation (60 trajectories)
                 Only use first EARLY_FRAC of each training trajectory for fine-tuning.
                 Test on FULL RUL across 10 held-out test trajectories.

For N in [1, 2, 3, 5, 10, 20, 40] training trajectories:
  Transfer  : pre-train on NASA → fine-tune head + unfreeze encoder on N × EARLY_FRAC windows
  Scratch   : train everything from scratch on same N × EARLY_FRAC windows
  Test      : predict RUL on all windows of 10 held-out test trajectories (full life)

Expected result: Transfer >> Scratch when N is small (1-5),
because the pre-trained encoder already knows "what degradation looks like"
and can extrapolate beyond the early-life window it has seen.

Usage:
  python scripts/exp_few_shot_transfer.py \\
      --nasa-dir  data/processed/nasa_battery/5. Battery Data Set \\
      --geo-csv   data/processed/geo_battery_60traj/geo_battery_60traj.csv \\
      --output    outputs/few_shot_transfer_v1 \\
      --device    cuda:1
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

from src.data.battery_strict import load_battery_mat, materialize_battery

# ── constants ─────────────────────────────────────────────────────────────────
SEEDS        = (42, 123, 456, 2026, 3407)
SEQ_LEN      = 16       # SOH window length (cycles)
HIDDEN       = 64
DROPOUT      = 0.1
EPOCHS_PRETRAIN  = 400  # pre-train on NASA
EPOCHS_FINETUNE  = 300  # fine-tune on early-life GEO
EPOCHS_SCRATCH   = 500  # scratch on early-life GEO (needs more time to converge)
BATCH_SIZE   = 256
LR           = 5e-4
EARLY_FRAC   = 0.20     # Only first 20% of trajectory life used for fine-tuning/scratch
               #  → simulates "satellite just launched, only early data available"
RUL_SCALE    = 1.0      # Use normalized RUL (0-1) as training target — domain-invariant

NASA_AUX = ("B0042", "B0043", "B0044", "B0046", "B0047", "B0048")
N_SHOTS  = (1, 2, 3, 5, 10, 20, 40)
N_TEST   = 10   # held-out GEO test trajectories

# ── simple shared model ───────────────────────────────────────────────────────
class SOHEncoder(nn.Module):
    """Encode a SOH window [L] → fixed-dim vector."""
    def __init__(self, seq_len=SEQ_LEN, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=2, batch_first=True,
                          dropout=dropout, bidirectional=False)
        self.norm = nn.LayerNorm(hidden)
        self.seq_len = seq_len
        self.hidden  = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 1]
        out, _ = self.gru(x)
        return self.norm(out[:, -1, :])   # last step


class RULHead(nn.Module):
    """Encoder output → RUL (raw cycles)."""
    def __init__(self, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(z).squeeze(-1))


class TransferModel(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.encoder = SOHEncoder(seq_len, hidden, dropout)
        self.head    = RULHead(hidden, dropout)

    def forward(self, x):
        return self.head(self.encoder(x))


# ── helpers ───────────────────────────────────────────────────────────────────
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else float("nan")

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


# ── data preparation ──────────────────────────────────────────────────────────
def soh_from_nasa(mat_path, protocol="rel80", seq_len=SEQ_LEN):
    """Extract (soh_windows, rul_normalized, max_rul) from one NASA battery."""
    raw = load_battery_mat(str(mat_path))
    series = materialize_battery(raw, protocol)
    if series.status != "event_observed":
        return None
    soh = series.soh.astype(np.float32)
    rul = series.rul.astype(np.float32)
    if len(soh) < seq_len + 1:
        return None
    max_rul = float(rul[0]) if rul[0] > 0 else float(np.nanmax(rul))
    if max_rul < 1:
        return None
    rul_norm = rul / max_rul   # normalize to 0-1 range
    windows, targets = [], []
    for end in range(seq_len - 1, len(soh)):
        windows.append(soh[end - seq_len + 1: end + 1])
        targets.append(rul_norm[end])
    X = np.array(windows, dtype=np.float32)[:, :, None]
    y = np.array(targets, dtype=np.float32)
    return X, y, max_rul


def soh_from_geo(df_traj, seq_len=SEQ_LEN, eol_soh=0.80,
                  early_frac=EARLY_FRAC):
    """Extract early-life and full windows from one GEO PINN trajectory.
    Returns (X_early, y_early, X_full, y_full, max_rul) — all RUL normalized 0-1.
    early-life windows = first EARLY_FRAC of trajectory (for few-shot training)
    full windows = entire trajectory (for test evaluation)
    """
    soh   = df_traj["soh"].values.astype(np.float32)
    cycle = df_traj["cycle"].values.astype(np.float32)
    hits  = np.flatnonzero(soh <= eol_soh)
    if hits.size == 0:
        return None
    eol_cycle = float(cycle[hits[0]])
    max_rul = eol_cycle
    if max_rul < 1:
        return None
    rul      = np.maximum(eol_cycle - cycle, 0.).astype(np.float32)
    rul_norm = rul / max_rul

    cutoff = max(int(len(soh) * early_frac), seq_len + 1)

    Xf, yf, Xe, ye = [], [], [], []
    for end in range(seq_len - 1, len(soh)):
        win = soh[end - seq_len + 1: end + 1]
        Xf.append(win); yf.append(rul_norm[end])
        if end < cutoff:
            Xe.append(win); ye.append(rul_norm[end])

    if not Xe or not Xf:
        return None
    return (np.array(Xe, dtype=np.float32)[:, :, None],
            np.array(ye, dtype=np.float32),
            np.array(Xf, dtype=np.float32)[:, :, None],
            np.array(yf, dtype=np.float32),
            max_rul)


# ── training ──────────────────────────────────────────────────────────────────
def make_tensors(X_list, y_list, device):
    X = torch.as_tensor(np.concatenate(X_list, axis=0), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.concatenate(y_list, axis=0), dtype=torch.float32, device=device)
    return X, y


def train_model(model, X, y, device, seed, epochs,
                freeze_encoder=False, patience=50):
    seed_all(seed)
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        params = list(model.head.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    batches = math.ceil(max(len(X), 1) / BATCH_SIZE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * batches))
    rng = np.random.default_rng(seed)
    best_loss, best_state, stale = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(len(X))
        ep_loss = 0.
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s: s + BATCH_SIZE]
            pred = model(X[b])
            target = y[b] / RUL_SCALE
            loss = F.smooth_l1_loss(pred, target, beta=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item() * len(b)
        ep_loss /= len(X)
        if ep_loss < best_loss - 1e-6:
            best_loss, stale = ep_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    # re-enable all gradients
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    preds = []
    for s in range(0, len(X), BATCH_SIZE):
        b = X[s: s + BATCH_SIZE]
        preds.append(model(b).cpu().numpy())
    return np.concatenate(preds) * RUL_SCALE


# ── main benchmark ────────────────────────────────────────────────────────────
def run_few_shot(nasa_dir, geo_csv_path, out_dir, device, quick=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:2] if quick else SEEDS
    epochs_pre  = 20 if quick else EPOCHS_PRETRAIN
    epochs_ft   = 10 if quick else EPOCHS_FINETUNE
    epochs_scr  = 20 if quick else EPOCHS_SCRATCH

    import pandas as pd
    geo_df = pd.read_csv(geo_csv_path)
    geo_traj_ids = geo_df["trajectory_id"].unique().tolist()

    # Fix test set: last N_TEST trajectories (deterministic)
    rng_split = np.random.default_rng(2026)
    perm = rng_split.permutation(len(geo_traj_ids))
    test_ids  = [geo_traj_ids[i] for i in perm[:N_TEST]]
    train_ids = [geo_traj_ids[i] for i in perm[N_TEST:]]
    print(f"GEO trajectories: {len(geo_traj_ids)} total, "
          f"{len(train_ids)} train pool, {len(test_ids)} test", flush=True)

    # Prepare GEO data: early-life for training, full for test evaluation
    # soh_from_geo returns (Xe, ye, Xf, yf, max_rul)
    geo_train_data = {}   # tid → (Xe, ye, max_rul)  ← early-life only
    for tid in train_ids:
        sub = geo_df[geo_df["trajectory_id"] == tid]
        d = soh_from_geo(sub, SEQ_LEN)
        if d is not None:
            geo_train_data[tid] = (d[0], d[1], d[4])   # Xe, ye_norm, max_rul

    # Test: full trajectory windows for evaluation
    geo_test_Xs, geo_test_ys_norm, geo_test_max_ruls = [], [], []
    for tid in test_ids:
        sub = geo_df[geo_df["trajectory_id"] == tid]
        d = soh_from_geo(sub, SEQ_LEN)
        if d is not None:
            geo_test_Xs.append(d[2])             # Xf = full windows
            geo_test_ys_norm.append(d[3])        # yf = full normalized RUL
            geo_test_max_ruls.append(np.full(len(d[3]), d[4], dtype=np.float32))

    if not geo_test_Xs:
        print("No usable GEO test trajectories!", flush=True)
        return {}
    Xtest_geo   = torch.as_tensor(np.concatenate(geo_test_Xs), dtype=torch.float32, device=device)
    ytest_norm  = np.concatenate(geo_test_ys_norm)      # 0-1 normalized
    ytest_max   = np.concatenate(geo_test_max_ruls)     # max_rul per window
    ytest_cycles = ytest_norm * ytest_max               # actual cycles
    print(f"GEO test: {len(Xtest_geo)} windows, "
          f"RUL cycle range {ytest_cycles.min():.0f}-{ytest_cycles.max():.0f}", flush=True)

    # Prepare NASA source data (normalized RUL)
    print("\nLoading NASA source data...", flush=True)
    nasa_Xs, nasa_ys = [], []
    for cell_name in NASA_AUX:
        mat = Path(nasa_dir) / f"{cell_name}.mat"
        if not mat.exists():
            continue
        d = soh_from_nasa(mat, protocol="rel80", seq_len=SEQ_LEN)
        if d is not None:
            nasa_Xs.append(d[0]); nasa_ys.append(d[1])   # y is already normalized
            print(f"  {cell_name}: {len(d[0])} windows  max_rul={d[2]:.0f}cycles", flush=True)

    if not nasa_Xs:
        print("No NASA data found!", flush=True)
        return {}
    Xnasa = torch.as_tensor(np.concatenate(nasa_Xs), dtype=torch.float32, device=device)
    ynasa = torch.as_tensor(np.concatenate(nasa_ys), dtype=torch.float32, device=device)
    print(f"NASA total: {len(Xnasa)} windows, norm RUL range "
          f"{ynasa.min().item():.3f}-{ynasa.max().item():.3f}  (should be ~0-1)", flush=True)

    # ── Step 1: Pre-train on NASA ────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Step 1: Pre-training encoder on NASA ({len(NASA_AUX)} cells, {epochs_pre} epochs)", flush=True)
    pretrain_states = []
    for seed in seeds:
        m = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
        m = train_model(m, Xnasa, ynasa, device, seed, epochs_pre)
        pretrain_states.append({k: v.cpu().clone() for k, v in m.state_dict().items()})
    print(f"Pre-training done.", flush=True)

    # ── Step 2: N-shot experiments ───────────────────────────────────────────
    results = {}
    n_shots = [1, 2] if quick else N_SHOTS

    for N in n_shots:
        avail = list(geo_train_data.keys())
        if len(avail) < N:
            print(f"  N={N}: not enough training trajectories ({len(avail)} avail)", flush=True)
            continue

        transfer_rmses, scratch_rmses = [], []

        for trial in range(3 if not quick else 2):
            rng_t = np.random.default_rng(trial * 1000 + N)
            chosen = rng_t.choice(len(avail), N, replace=False)
            sel_ids = [avail[i] for i in chosen]
            Xs = [geo_train_data[t][0] for t in sel_ids]
            ys = [geo_train_data[t][1] for t in sel_ids]
            Xfew = torch.as_tensor(np.concatenate(Xs), dtype=torch.float32, device=device)
            yfew = torch.as_tensor(np.concatenate(ys), dtype=torch.float32, device=device)
            n_early_windows = sum(len(geo_train_data[t][0]) for t in sel_ids)
            print(f"  N={N} trial={trial}: {n_early_windows} early-life windows "
                  f"(first {int(EARLY_FRAC*100)}% of life) from {sel_ids}", flush=True)

            for seed_idx, seed in enumerate(seeds):
                # Transfer: load pre-trained encoder, fine-tune head only
                m_transfer = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                m_transfer.load_state_dict(pretrain_states[seed_idx])
                m_transfer = train_model(m_transfer, Xfew, yfew, device, seed,
                                         epochs_ft, freeze_encoder=True)
                # Predict normalized RUL, then de-normalize to cycles
                tp_norm = predict(m_transfer, Xtest_geo, device)
                tp_cycles = tp_norm * ytest_max          # de-normalize
                transfer_rmses.append(rmse(ytest_cycles, tp_cycles))

                # Scratch: random init, train from scratch
                m_scratch = TransferModel(SEQ_LEN, HIDDEN, DROPOUT).to(device)
                m_scratch = train_model(m_scratch, Xfew, yfew, device, seed,
                                        epochs_scr, freeze_encoder=False)
                sp_norm = predict(m_scratch, Xtest_geo, device)
                sp_cycles = sp_norm * ytest_max
                scratch_rmses.append(rmse(ytest_cycles, sp_cycles))

        t_mean = float(np.mean(transfer_rmses))
        s_mean = float(np.mean(scratch_rmses))
        gain   = (s_mean - t_mean) / max(s_mean, 1e-9) * 100
        results[N] = {
            "n_shot": N,
            "transfer_mean_rmse": t_mean,
            "scratch_mean_rmse":  s_mean,
            "transfer_rmses": transfer_rmses,
            "scratch_rmses":  scratch_rmses,
            "gain_pct": round(gain, 2),
        }
        print(f"  N={N:3d}: Transfer={t_mean:.2f}  Scratch={s_mean:.2f}  "
              f"Transfer advantage: +{gain:.1f}%", flush=True)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("FEW-SHOT TRANSFER LEARNING RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'N':>5} {'Transfer RMSE':>16} {'Scratch RMSE':>15} {'Gain':>8}", flush=True)
    print(f"{'-'*50}", flush=True)
    for N, r in sorted(results.items()):
        print(f"{N:>5} {r['transfer_mean_rmse']:>16.2f} {r['scratch_mean_rmse']:>15.2f} {r['gain_pct']:>7.1f}%",
              flush=True)
    print(f"{'='*60}", flush=True)

    result_obj = {
        "schema":       "few_shot_transfer_v1",
        "source_domain": "NASA battery (B0042-B0048, rel80)",
        "target_domain": "GEO satellite PINN simulation (60 trajectories)",
        "feature":      "SOH window (shared degradation signal)",
        "seq_len":      SEQ_LEN,
        "n_test_traj":  N_TEST,
        "n_test_windows": int(len(ytest_cycles)),
        "seeds":        list(seeds),
        "epochs_pretrain": epochs_pre,
        "epochs_finetune": epochs_ft,
        "epochs_scratch":  epochs_scr,
        "results":      results,
    }
    write_json(out_dir / "FEW_SHOT_TRANSFER_REPORT.json", result_obj)

    # Save CSV for plotting
    rows = [{"n_shot": N, "method": m, "rmse": r}
            for N, res in results.items()
            for m, r in [("transfer", res["transfer_mean_rmse"]),
                         ("scratch",  res["scratch_mean_rmse"])]]
    with (out_dir / "few_shot_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["n_shot", "method", "rmse"])
        w.writeheader(); w.writerows(rows)

    print(f"\nReport → {out_dir}/FEW_SHOT_TRANSFER_REPORT.json", flush=True)
    return result_obj


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nasa-dir", default="data/processed/nasa_battery/5. Battery Data Set")
    parser.add_argument("--geo-csv",  default="data/processed/geo_battery_60traj/geo_battery_60traj.csv")
    parser.add_argument("--output",   default="outputs/few_shot_transfer_v1")
    parser.add_argument("--device",   default="cuda:1")
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Few-shot transfer  device={device}  quick={args.quick}", flush=True)

    t0 = time.time()
    run_few_shot(
        nasa_dir=args.nasa_dir,
        geo_csv_path=args.geo_csv,
        out_dir=Path(args.output),
        device=device,
        quick=args.quick,
    )
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
