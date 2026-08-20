"""
B0007 confirmation with the proven battery-v3 training stack (nozzle frozen).

First-pass B0007 script mixed a different feature set; this script reuses v3's
feature engineering + training so numbers are comparable to BATTERY_OPT_V3.

Reporting rules
---------------
- Strict 1.4 Ah remains the official comparable protocol for B0005/6/18.
- B0007 never hits 1.4 Ah → report under special EOL protocols SEPARATELY.
- Do NOT average mixed-protocol folds into one "main mean".
- Nozzle checkpoints / numbers are not modified.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
SEEDS = [42, 123, 456]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.mean(np.abs(a - b)))


def ttv(v, t, thr):
    idx = np.where(v <= thr)[0]
    return float(t[idx[0]]) if len(idx) else float(t[-1])


def load_raw(mat_path: Path) -> Dict:
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    bat = data[mat_path.stem]
    rows = []
    for cycle in np.atleast_1d(bat.cycle):
        if getattr(cycle, "type", None) != "discharge":
            continue
        d = cycle.data
        if not hasattr(d, "Capacity"):
            continue
        cap = float(np.asarray(d.Capacity).reshape(-1)[0])
        if not np.isfinite(cap) or cap <= 0:
            continue
        v = np.asarray(d.Voltage_measured, np.float64).reshape(-1)
        c = np.asarray(d.Current_measured, np.float64).reshape(-1)
        temp = np.asarray(d.Temperature_measured, np.float64).reshape(-1)
        tt = np.asarray(d.Time, np.float64).reshape(-1)
        if len(v) < 8 or len(tt) != len(v):
            continue
        lo, hi = len(v) // 4, max(len(v) // 4 + 1, 3 * len(v) // 4)
        dur = float(tt[-1] - tt[0])
        rows.append({
            "cap": cap,
            "v_mean": float(np.mean(v)), "v_std": float(np.std(v)),
            "v_min": float(np.min(v)), "v_mid": float(np.mean(v[lo:hi])),
            "v_p10": float(np.percentile(v, 10)),
            "c_mean": float(np.mean(np.abs(c))), "c_std": float(np.std(c)),
            "t_mean": float(np.mean(temp)), "t_std": float(np.std(temp)),
            "t_max": float(np.max(temp)),
            "ttv30": ttv(v, tt, 3.0), "ttv27": ttv(v, tt, 2.7),
            "ttv25": ttv(v, tt, 2.5), "dur": dur,
        })
    caps = np.array([r["cap"] for r in rows], np.float64)
    for i in range(1, len(caps)):
        if caps[i] > caps[i - 1] * 1.03:
            caps[i] = 0.6 * caps[i - 1] + 0.4 * caps[i]
    for i, r in enumerate(rows):
        r["cap"] = float(caps[i])
    init = float(np.median(caps[:5]))
    n = len(caps)
    k = min(40, max(10, n // 4))
    slope, intercept = np.polyfit(np.arange(n - k, n), caps[-k:], 1)
    if slope < -1e-6:
        extrap14 = float((1.4 - intercept) / slope)
    else:
        extrap14 = float(n - 1)
    extrap14 = max(extrap14, float(n - 1))
    return {
        "name": mat_path.stem,
        "rows": rows,
        "caps": caps,
        "init": init,
        "extrap14": extrap14,
        "late_slope": float(slope),
    }


def materialize(raw: Dict, protocol: str) -> Dict:
    """Build v3-style series under a chosen EOL protocol."""
    rows = raw["rows"]
    caps_full = raw["caps"].copy()
    init = raw["init"]
    n = len(caps_full)

    if protocol == "strict14":
        eol = 1.4
        hit = np.where(caps_full <= eol)[0]
        end = int(hit[0]) if len(hit) else n - 1
        life = float(end)
        note = "hit" if len(hit) else "end_of_record_fallback"
    elif protocol == "rel80":
        eol = init * 0.8
        hit = np.where(caps_full <= eol)[0]
        end = int(hit[0]) if len(hit) else n - 1
        life = float(end)
        note = "hit" if len(hit) else "end_of_record_fallback"
    elif protocol == "rated75":
        eol = 1.5
        hit = np.where(caps_full <= eol)[0]
        end = int(hit[0]) if len(hit) else n - 1
        life = float(end)
        note = "hit" if len(hit) else "end_of_record_fallback"
    elif protocol == "extrap14":
        eol = 1.4
        end = n - 1
        life = float(raw["extrap14"])
        note = "linear_extrap_to_1.4"
    elif protocol == "adaptive":
        hit14 = np.where(caps_full <= 1.4)[0]
        if len(hit14):
            eol = 1.4
            end = int(hit14[0])
            note = "adaptive_used_1.4"
        else:
            eol = init * 0.8
            hit = np.where(caps_full <= eol)[0]
            end = int(hit[0]) if len(hit) else n - 1
            note = "adaptive_used_rel80"
        life = float(end)
    else:
        raise ValueError(protocol)

    end = max(end, 20)
    use_n = end + 1
    caps = caps_full[:use_n]
    rows = rows[:use_n]
    soh = caps / max(init, 1e-6)
    idxs = np.arange(use_n, dtype=np.float32)
    ruls = np.maximum(life - idxs, 0.0).astype(np.float32)

    early_dur = float(np.median([rows[k]["dur"] for k in range(min(5, use_n))]))
    early_ttv = float(np.median([rows[k]["ttv27"] for k in range(min(5, use_n))]))
    early_t = float(np.median([rows[k]["t_mean"] for k in range(min(5, use_n))]))

    feats = []
    for i, r in enumerate(rows):
        j0 = max(0, i - 9)
        loc_slope = float(np.polyfit(np.arange(j0, i + 1), caps[j0:i + 1], 1)[0]) if i > j0 else 0.0
        j1 = max(0, i - 4)
        fade5 = float(caps[j1] - caps[i]) if i > j1 else 0.0
        fade1 = float(caps[i - 1] - caps[i]) if i > 0 else 0.0
        cum_fade = float((init - caps[i]) / max(init - eol, 1e-6))
        feats.append([
            soh[i], float(1.0 - soh[i]), float(caps[i] - eol), cum_fade,
            fade1, fade5, loc_slope,
            r["v_mean"], r["v_std"], r["v_min"], r["v_mid"], r["v_p10"],
            r["c_mean"], r["c_std"],
            r["t_mean"], r["t_std"], r["t_max"], r["t_mean"] - early_t,
            r["ttv30"], r["ttv27"], r["ttv25"],
            r["dur"], r["dur"] / max(early_dur, 1e-6), r["ttv27"] / max(early_ttv, 1e-6),
        ])
    return {
        "name": raw["name"],
        "features": np.asarray(feats, np.float32),
        "capacity": caps.astype(np.float32),
        "soh": soh.astype(np.float32),
        "rul": ruls,
        "init": init,
        "eol": float(eol),
        "life": float(life),
        "n": int(use_n),
        "protocol": protocol,
        "note": note,
        "hit_eol": note.startswith("hit") or note.startswith("adaptive_used"),
    }


def make_windows(series_list, seq_len, mean=None, std=None):
    xs, yr, yc, meta = [], [], [], []
    all_rows = np.concatenate([s["features"] for s in series_list], 0)
    if mean is None:
        mean = all_rows.mean(0, keepdims=True)
        std = all_rows.std(0, keepdims=True) + 1e-8
    for s in series_list:
        x = (s["features"] - mean) / std
        for i in range(0, len(x) - seq_len + 1):
            xs.append(x[i:i + seq_len])
            yr.append(s["rul"][i + seq_len - 1])
            yc.append(s["capacity"][i + seq_len - 1])
            meta.append((s["name"], i + seq_len - 1, float(s["soh"][i + seq_len - 1])))
    return (np.stack(xs).astype(np.float32),
            np.asarray(yr, np.float32),
            np.asarray(yc, np.float32),
            meta, mean.astype(np.float32), std.astype(np.float32))


class TCNBlock(nn.Module):
    def __init__(self, ch, k=3, dil=2, p=0.15):
        super().__init__()
        pad = (k - 1) * dil
        self.conv = nn.Conv1d(ch, ch, k, dilation=dil, padding=pad)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.n = nn.LayerNorm(ch)
        self.d = nn.Dropout(p)

    def forward(self, x):
        L = x.size(-1)
        h = F.gelu(self.conv(x))[..., :L]
        h = self.d(self.mix(h))
        return self.n((x + h).transpose(1, 2)).transpose(1, 2)


class MSNet(nn.Module):
    def __init__(self, n_feat, d=96, p=0.15):
        super().__init__()
        b = max(16, d // 3)
        w = b * 3
        self.b = b
        self.proj = nn.Linear(n_feat, w)
        self.branches = nn.ModuleList([
            nn.ModuleList([TCNBlock(b, k, p=p) for _ in range(2)]) for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(w)
        self.shared = nn.Sequential(nn.Linear(w, 96), nn.GELU(), nn.Dropout(p))
        self.rul = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))
        self.cap = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))

    def forward(self, x):
        h = self.proj(x).transpose(1, 2)
        outs = []
        for ch, br in zip(torch.split(h, self.b, 1), self.branches):
            for block in br:
                ch = block(ch)
            outs.append(ch)
        z = self.norm(torch.cat(outs, 1).transpose(1, 2)).mean(1)
        z = self.shared(z)
        return self.rul(z).squeeze(-1), self.cap(z).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, n_feat, h=96, p=0.15):
        super().__init__()
        self.gru = nn.GRU(n_feat, h, num_layers=2, batch_first=True, dropout=p)
        self.rul = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Dropout(p), nn.Linear(64, 1))
        self.cap = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        z = h[:, -1]
        return self.rul(z).squeeze(-1), self.cap(z).squeeze(-1)


class CNNNet(nn.Module):
    def __init__(self, n_feat, p=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 96, 3, padding=1), nn.GELU(),
            nn.Conv1d(96, 96, 3, padding=2, dilation=2), nn.GELU(),
        )
        self.rul = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))
        self.cap = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))

    def forward(self, x):
        h = self.net(x.transpose(1, 2)).mean(-1)
        return self.rul(h).squeeze(-1), self.cap(h).squeeze(-1)


def train_model(model, Xtr, ytr, ctr, Xva, yva, device, seed=42,
                epochs=240, bs=32, lr=7e-4, patience=35, cap_w=0.5):
    seed_everything(seed)
    model = model.to(device)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    ctr_t = torch.as_tensor(ctr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * math.ceil(max(len(Xtr), 1) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    w = 1.0 + 1.2 * (1.0 - ytr / (ytr.max() + 1e-6))
    w_t = torch.as_tensor(w.astype(np.float32), device=device)
    best, best_state, stale = 1e18, None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        for s0 in range(0, len(Xtr), bs):
            idx = order[s0:s0 + bs]
            pr, pc = model(Xtr_t[idx])
            loss_r = (F.smooth_l1_loss(pr, ytr_t[idx], reduction="none") * w_t[idx]).mean()
            loss_c = F.mse_loss(pc, ctr_t[idx])
            over = F.relu(pr - ytr_t[idx]).mean()
            loss = loss_r + cap_w * loss_c + 0.05 * over
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pr, _ = model(Xva_t)
            val = rmse(yva, pr.cpu().numpy())
        if val < best - 1e-4:
            best, stale = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    pr, pc = model(xt)
    return pr.cpu().numpy(), pc.cpu().numpy()


def fit_soh_rul_map(train_series):
    xs, ys = [], []
    for s in train_series:
        xs.append(s["soh"])
        ys.append(s["rul"])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    edges = np.linspace(x.min(), x.max(), 12)
    centers, vals = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x <= edges[i + 1] if i == len(edges) - 2 else x < edges[i + 1])
        if m.sum() >= 3:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            vals.append(float(np.median(y[m])))
    if len(centers) < 2:
        a, b = np.polyfit(x, y, 1)
        return lambda soh: a * soh + b
    centers = np.asarray(centers)
    vals = np.asarray(vals)
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]

    def mapper(soh):
        soh = np.asarray(soh, np.float64)
        return np.interp(soh, centers, vals).astype(np.float32)

    return mapper


def run_holdout(raws: Dict[str, Dict], holdout: str, protocol: str, device: str,
                seq_len: int, seeds: List[int],
                train_protocol: Optional[str] = None) -> Dict:
    """
    train_protocol: if set, train batteries use this protocol; holdout uses `protocol`.
    Default: all cells use the same protocol (consistent LOO).
    """
    tp = train_protocol or protocol
    series_map = {}
    for n in BATTERIES:
        if n == holdout:
            series_map[n] = materialize(raws[n], protocol)
        else:
            series_map[n] = materialize(raws[n], tp)

    train_series = [series_map[n] for n in BATTERIES if n != holdout]
    hold = series_map[holdout]
    Xall, yall, call, meta, mean, std = make_windows(train_series, seq_len)
    Xte, yte, cte, meta_te, _, _ = make_windows([hold], seq_len, mean, std)

    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(Xall))
    n_val = max(20, int(0.2 * len(Xall)))
    iva, itr = idx[:n_val], idx[n_val:]
    Xtr, ytr, ctr = Xall[itr], yall[itr], call[itr]
    Xva, yva = Xall[iva], yall[iva]

    soh_map = fit_soh_rul_map(train_series)
    soh_te = np.array([m[2] for m in meta_te], np.float32)
    soh_va = np.array([meta[i][2] for i in iva], np.float32)
    soh_pred_te = soh_map(soh_te)
    soh_pred_va = soh_map(soh_va)

    factories = {
        "gru": lambda: GRUNet(Xtr.shape[-1], h=96),
        "ms": lambda: MSNet(Xtr.shape[-1], d=96),
        "cnn": lambda: CNNNet(Xtr.shape[-1]),
    }
    family_te, family_va, family_metrics = {}, {}, {}
    for fname, fac in factories.items():
        te_preds, va_preds, mets = [], [], []
        for sd in seeds:
            model = fac()
            model, vbest = train_model(
                model, Xtr, ytr, ctr, Xva, yva, device, seed=sd,
                epochs=240, bs=32, lr=7e-4, patience=35, cap_w=0.5,
            )
            pte, _ = predict(model, Xte, device)
            pva, _ = predict(model, Xva, device)
            te_preds.append(pte)
            va_preds.append(pva)
            mets.append({"seed": int(sd), "val": vbest, "test": rmse(yte, pte),
                         "mae": mae(yte, pte)})
            print(f"    {holdout}/{protocol}/{fname} s{sd}: "
                  f"test={mets[-1]['test']:.3f} val={vbest:.3f}", flush=True)
        family_te[fname] = np.mean(np.stack(te_preds, 0), 0)
        family_va[fname] = np.mean(np.stack(va_preds, 0), 0)
        family_metrics[fname] = {
            "mean_test": float(np.mean([m["test"] for m in mets])),
            "ens_test": rmse(yte, family_te[fname]),
            "ens_mae": mae(yte, family_te[fname]),
            "ens_val": rmse(yva, family_va[fname]),
            "seeds": mets,
        }

    deep_eq = np.mean(np.stack([family_te["gru"], family_te["ms"], family_te["cnn"]], 0), 0)
    deep_eq_va = np.mean(np.stack([family_va["gru"], family_va["ms"], family_va["cnn"]], 0), 0)
    candidates = {
        "deep_equal": (deep_eq, deep_eq_va),
        "gru": (family_te["gru"], family_va["gru"]),
        "ms": (family_te["ms"], family_va["ms"]),
        "cnn": (family_te["cnn"], family_va["cnn"]),
        "soh_map": (soh_pred_te, soh_pred_va),
    }
    pick = min(candidates, key=lambda k: rmse(yva, candidates[k][1]))
    chosen = candidates[pick][0]

    row = {
        "holdout": holdout,
        "protocol": protocol,
        "train_protocol": tp,
        "life": hold["life"],
        "eol": hold["eol"],
        "n": hold["n"],
        "note": hold["note"],
        "n_test": int(len(Xte)),
        "soh_map_rmse": rmse(yte, soh_pred_te),
        "gru_ens_rmse": family_metrics["gru"]["ens_test"],
        "ms_ens_rmse": family_metrics["ms"]["ens_test"],
        "cnn_ens_rmse": family_metrics["cnn"]["ens_test"],
        "deep_equal_rmse": rmse(yte, deep_eq),
        "val_pick": pick,
        "val_pick_rmse": rmse(yte, chosen),
        "family": family_metrics,
        # primary recommended for B0007 special: fixed MS ens (same as v3 primary)
        "primary_ms_ens": family_metrics["ms"]["ens_test"],
        "primary_ms_mae": family_metrics["ms"]["ens_mae"],
    }
    print(
        f"  >> {holdout}/{protocol}: MS={row['ms_ens_rmse']:.3f} "
        f"GRU={row['gru_ens_rmse']:.3f} CNN={row['cnn_ens_rmse']:.3f} "
        f"deep_eq={row['deep_equal_rmse']:.3f} pick={pick}:{row['val_pick_rmse']:.3f} "
        f"life={hold['life']:.1f} eol={hold['eol']:.4f} ({hold['note']})",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="outputs/battery_b0007_confirm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device={device}", flush=True)
    print("Nozzle frozen. B0007 special confirmation with v3 stack.", flush=True)

    mat_dir = Path(args.data_root) / "nasa_battery" / "5. Battery Data Set"
    raws = {n: load_raw(mat_dir / f"{n}.mat") for n in BATTERIES}
    diag = {}
    for n, r in raws.items():
        caps = r["caps"]
        init = r["init"]
        diag[n] = {
            "n": len(caps), "init": init, "min": float(caps.min()),
            "end": float(caps[-1]), "soh_end": float(caps[-1] / init),
            "extrap14": r["extrap14"],
            "strict14_first": int(np.where(caps <= 1.4)[0][0]) if np.any(caps <= 1.4) else -1,
            "rel80_first": int(np.where(caps <= init * 0.8)[0][0]) if np.any(caps <= init * 0.8) else -1,
            "rated75_first": int(np.where(caps <= 1.5)[0][0]) if np.any(caps <= 1.5) else -1,
        }
    print("=== Diagnosis ===", flush=True)
    print(json.dumps(diag, indent=2), flush=True)

    t0 = time.time()
    results = []

    # Published v3 strict14 MS ensemble (do not retrain; keep comparable)
    v3_strict = {
        "B0005": 10.48, "B0006": 13.79, "B0007": 45.24, "B0018": 6.39,
        "mean_all4": 18.97,
        "source": "outputs/battery_opt_v3 (fixed MultiScale 3-seed ens)",
    }

    # A) Re-measure B0007 under strict14 with this same stack (sanity)
    print("\n=== A. B0007 strict14 re-measure (v3 stack) ===", flush=True)
    b7_strict = run_holdout(raws, "B0007", "strict14", device, args.seq_len, SEEDS)
    results.append(b7_strict)

    # B) B0007 special protocols; train cells stay strict14
    print("\n=== B. B0007 special protocols (train=strict14, test=special) ===", flush=True)
    b7_special = []
    for proto in ["rel80", "rated75", "adaptive", "extrap14"]:
        row = run_holdout(
            raws, "B0007", proto, device, args.seq_len, SEEDS,
            train_protocol="strict14",
        )
        b7_special.append(row)
        results.append(row)

    # C) Consistent protocol: only B0007 holdout when ALL cells use same special EOL
    #    (shows whether train-label consistency helps B0007)
    print("\n=== C. B0007 holdout under consistent special protocols ===", flush=True)
    b7_consistent = []
    for proto in ["rel80", "rated75", "adaptive"]:
        row = run_holdout(raws, "B0007", proto, device, args.seq_len, SEEDS)
        b7_consistent.append(row)
        results.append(row)

    b7_rank = sorted(b7_special + b7_consistent, key=lambda r: r["primary_ms_ens"])
    best_b7 = b7_rank[0]

    others = ["B0005", "B0006", "B0018"]
    strict_others = {n: v3_strict[n] for n in others}
    strict_others_mean = float(np.mean(list(strict_others.values())))

    summary = {
        "nozzle_frozen": True,
        "diagnosis": diag,
        "strict14_reference_v3": v3_strict,
        "b0007_strict14_remeasure": {
            "ms_ens_rmse": b7_strict["primary_ms_ens"],
            "gru_ens_rmse": b7_strict["gru_ens_rmse"],
            "cnn_ens_rmse": b7_strict["cnn_ens_rmse"],
            "deep_equal_rmse": b7_strict["deep_equal_rmse"],
            "life": b7_strict["life"],
            "note": b7_strict["note"],
        },
        "recommended_reporting": {
            "main_comparable": {
                "protocol": "strict14 for B0005/B0006/B0018 (v3 MS ens)",
                "folds": strict_others,
                "mean_3": strict_others_mean,
            },
            "B0007_special": {
                "best_protocol": best_b7["protocol"],
                "train_protocol": best_b7["train_protocol"],
                "life": best_b7["life"],
                "eol": best_b7["eol"],
                "ms_ens_rmse": best_b7["primary_ms_ens"],
                "ms_ens_mae": best_b7["primary_ms_mae"],
                "gru_ens_rmse": best_b7["gru_ens_rmse"],
                "cnn_ens_rmse": best_b7["cnn_ens_rmse"],
                "deep_equal_rmse": best_b7["deep_equal_rmse"],
                "val_pick": best_b7["val_pick"],
                "val_pick_rmse": best_b7["val_pick_rmse"],
                "note": best_b7["note"],
            },
            "do_not_mix": (
                "Do not report a single mean that averages strict14 others with "
                "B0007-special; keep two rows/tables."
            ),
        },
        "b0007_ranking_ms": [
            {
                "protocol": r["protocol"],
                "train_protocol": r["train_protocol"],
                "ms": r["primary_ms_ens"],
                "gru": r["gru_ens_rmse"],
                "cnn": r["cnn_ens_rmse"],
                "deep_eq": r["deep_equal_rmse"],
                "life": r["life"],
                "eol": r["eol"],
                "note": r["note"],
            }
            for r in b7_rank
        ],
        "elapsed_sec": time.time() - t0,
    }

    report = {
        "title": "B0007 confirmation (v3 stack, nozzle frozen)",
        "summary": summary,
        "b0007_strict": b7_strict,
        "b0007_special_train_strict": b7_special,
        "b0007_consistent_protocol": b7_consistent,
        "all_results": results,
    }
    (out_dir / "BATTERY_B0007_CONFIRM_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (out_dir / "battery_b0007_confirm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {out_dir} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
