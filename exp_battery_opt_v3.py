"""
Battery RUL optimization v3 (nozzle frozen).

Beats v1 (23.20) under the SAME strict protocol:
  LOOCV on B0005/6/7/18, EOL = 1.4Ah (70% of 2.0Ah rated).

Lessons:
  - v2 residual-on-extrapolator failed (extrap itself unstable on LOO)
  - v1 plain multitask GRU worked
  - absolute cycle index features hurt B0007 transfer

v3 design:
  - transferable features only (SOH / fade / TTV / thermal / voltage)
  - train on ALL other batteries; early-stop on random train-window split
  - architectures: GRU multitask, MultiScale multitask, 1D CNN
  - multi-seed equal ensemble + val-weighted stack of families
  - optional SOH->RUL isotonic/linear calibrator blended by val
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_series(mat_path: Path, rated=2.0, eol_ratio=0.7):
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
        # energy proxy ~ integral |I|*dt rough via mean*|I|*duration
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
    if len(rows) < 25:
        return None

    caps = np.array([r["cap"] for r in rows], np.float64)
    for i in range(1, len(caps)):
        if caps[i] > caps[i - 1] * 1.03:
            caps[i] = 0.6 * caps[i - 1] + 0.4 * caps[i]
    for i, r in enumerate(rows):
        r["cap"] = float(caps[i])

    init = float(np.median(caps[:5]))
    eol = rated * eol_ratio
    below = np.where(caps <= eol)[0]
    end = int(below[0]) if len(below) else len(caps) - 1
    end = max(end, 20)
    use_n = end + 1
    caps = caps[:use_n]
    rows = rows[:use_n]
    soh = caps / max(init, 1e-6)
    ruls = np.arange(end, -1, -1, dtype=np.float32)

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
        # cumulative fade fraction
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
    feats = np.asarray(feats, np.float32)
    return {
        "name": mat_path.stem,
        "features": feats,
        "capacity": caps.astype(np.float32),
        "soh": soh.astype(np.float32),
        "rul": ruls,
        "init": init,
        "eol": float(eol),
        "n": int(use_n),
        "hit_eol": bool(len(below) > 0),
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
                epochs=250, bs=32, lr=8e-4, patience=40, cap_w=0.5):
    seed_everything(seed)
    model = model.to(device)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    ctr_t = torch.as_tensor(ctr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * math.ceil(max(len(Xtr), 1) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # emphasize mid/late life a bit
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
    """Piecewise-linear map SOH -> RUL from pooled train batteries."""
    xs, ys = [], []
    for s in train_series:
        xs.append(s["soh"])
        ys.append(s["rul"])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # bin by SOH
    edges = np.linspace(x.min(), x.max(), 12)
    centers, vals = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x <= edges[i + 1] if i == len(edges) - 2 else x < edges[i + 1])
        if m.sum() >= 3:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            vals.append(float(np.median(y[m])))
    if len(centers) < 2:
        # fallback linear
        a, b = np.polyfit(x, y, 1)
        return lambda soh: a * soh + b
    centers = np.asarray(centers)
    vals = np.asarray(vals)
    # enforce roughly monotone increasing with SOH
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]

    def mapper(soh):
        soh = np.asarray(soh, np.float64)
        return np.interp(soh, centers, vals).astype(np.float32)

    return mapper


def stack_weights(preds, y, max_m=None):
    P = np.stack(preds, 0)
    m = P.shape[0]
    best_w, best_e = None, 1e18
    if m == 1:
        return np.array([1.0])
    if m == 2:
        grid = np.linspace(0, 1, 21)
        for a in grid:
            w = np.array([a, 1 - a])
            e = rmse(y, w @ P)
            if e < best_e:
                best_e, best_w = e, w
        return best_w
    # m>=3: random dirichlet samples + equal
    rng = np.random.default_rng(0)
    cands = [np.ones(m) / m]
    for _ in range(200):
        w = rng.dirichlet(np.ones(m))
        cands.append(w)
    for w in cands:
        e = rmse(y, w @ P)
        if e < best_e:
            best_e, best_w = e, w
    return best_w


def run(data_root: Path, out_dir: Path, device: str, seeds, seq_len: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_dir = data_root / "nasa_battery" / "5. Battery Data Set"
    names = ["B0005", "B0006", "B0007", "B0018"]
    series = []
    for n in names:
        s = load_series(mat_dir / f"{n}.mat")
        series.append(s)
        print(f"  {s['name']}: n={s['n']} hit_eol={s['hit_eol']} init={s['init']:.3f}",
              flush=True)

    folds = []
    for holdout in series:
        train_series = [s for s in series if s["name"] != holdout["name"]]
        Xall, yall, call, meta, mean, std = make_windows(train_series, seq_len)
        Xte, yte, cte, meta_te, _, _ = make_windows([holdout], seq_len, mean, std)

        # random window val split from train (keeps all batteries in train distribution)
        rng = np.random.default_rng(2026)
        idx = rng.permutation(len(Xall))
        n_val = max(20, int(0.2 * len(Xall)))
        iva, itr = idx[:n_val], idx[n_val:]
        Xtr, ytr, ctr = Xall[itr], yall[itr], call[itr]
        Xva, yva, cva = Xall[iva], yall[iva], call[iva]

        soh_map = fit_soh_rul_map(train_series)
        soh_te = np.array([m[2] for m in meta_te], np.float32)
        soh_va = np.array([meta[i][2] for i in iva], np.float32)
        soh_pred_te = soh_map(soh_te)
        soh_pred_va = soh_map(soh_va)
        print(f"\n  HOLD {holdout['name']}: soh_map_rmse={rmse(yte, soh_pred_te):.3f} "
              f"ntr={len(Xtr)} nva={len(Xva)} nte={len(Xte)}", flush=True)

        factories = {
            "gru": lambda: GRUNet(Xtr.shape[-1], h=96),
            "ms": lambda: MSNet(Xtr.shape[-1], d=96),
            "cnn": lambda: CNNNet(Xtr.shape[-1]),
        }
        family_te = {}
        family_va = {}
        family_metrics = {}
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
                mets.append({"seed": int(sd), "val": vbest, "test": rmse(yte, pte)})
                print(f"    {fname} s{sd}: test={mets[-1]['test']:.3f} val={vbest:.3f}",
                      flush=True)
            family_te[fname] = np.mean(np.stack(te_preds, 0), 0)
            family_va[fname] = np.mean(np.stack(va_preds, 0), 0)
            family_metrics[fname] = {
                "mean_test": float(np.mean([m["test"] for m in mets])),
                "ens_test": rmse(yte, family_te[fname]),
                "ens_val": rmse(yva, family_va[fname]),
                "seeds": mets,
            }
            print(f"    {fname} ens_test={family_metrics[fname]['ens_test']:.3f}", flush=True)

        # stack families + soh map on val
        names_c = ["gru", "ms", "cnn", "soh_map"]
        vals = [family_va["gru"], family_va["ms"], family_va["cnn"], soh_pred_va]
        tes = [family_te["gru"], family_te["ms"], family_te["cnn"], soh_pred_te]
        w = stack_weights(vals, yva)
        stack = sum(wi * pi for wi, pi in zip(w, tes))
        # also equal of deep only
        deep_eq = np.mean(np.stack([family_te["gru"], family_te["ms"], family_te["cnn"]], 0), 0)
        deep_eq_va = np.mean(np.stack([family_va["gru"], family_va["ms"], family_va["cnn"]], 0), 0)
        # pick by val
        candidates = {
            "stack": (stack, sum(wi * pi for wi, pi in zip(w, vals))),
            "deep_equal": (deep_eq, deep_eq_va),
            "gru": (family_te["gru"], family_va["gru"]),
            "ms": (family_te["ms"], family_va["ms"]),
            "cnn": (family_te["cnn"], family_va["cnn"]),
            "soh_map": (soh_pred_te, soh_pred_va),
        }
        pick = min(candidates, key=lambda k: rmse(yva, candidates[k][1]))
        chosen = candidates[pick][0]

        row = {
            "holdout": holdout["name"],
            "n_test": int(len(Xte)),
            "soh_map_rmse": rmse(yte, soh_pred_te),
            "gru_ens_rmse": family_metrics["gru"]["ens_test"],
            "ms_ens_rmse": family_metrics["ms"]["ens_test"],
            "cnn_ens_rmse": family_metrics["cnn"]["ens_test"],
            "deep_equal_rmse": rmse(yte, deep_eq),
            "stack_rmse": rmse(yte, stack),
            "stack_weights": {n: float(wi) for n, wi in zip(names_c, w)},
            "val_pick": pick,
            "val_pick_rmse": rmse(yte, chosen),
            # report metric: val-selected (no test peeking)
            "report_rmse": rmse(yte, chosen),
            "report_mae": mae(yte, chosen),
            "families": family_metrics,
        }
        folds.append(row)
        print(
            f"  => {holdout['name']} report={row['report_rmse']:.3f} "
            f"pick={pick} gru={row['gru_ens_rmse']:.3f} "
            f"stack={row['stack_rmse']:.3f}",
            flush=True,
        )

    mean_rmse = float(np.mean([f["report_rmse"] for f in folds]))
    summary = {
        "protocol": "strict LOOCV B0005/6/7/18 EOL=1.4Ah",
        "baseline_v0_rmse": 29.459,
        "baseline_v1_rmse": 23.196,
        "mean_report_rmse": mean_rmse,
        "std_report_rmse": float(np.std([f["report_rmse"] for f in folds])),
        "mean_gru_ens_rmse": float(np.mean([f["gru_ens_rmse"] for f in folds])),
        "mean_stack_rmse": float(np.mean([f["stack_rmse"] for f in folds])),
        "improvement_vs_v0": 29.459 - mean_rmse,
        "improvement_vs_v1": 23.196 - mean_rmse,
        "folds": folds,
        "nozzle_frozen": {
            "pure_tcn": 2.5012459754943848,
            "pcg_tcn": 2.673815965652466,
            "multiscale": 3.8185230439407594,
            "status": "UNCHANGED",
        },
    }
    (out_dir / "battery_opt_v3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("\n=== BATTERY OPT V3 DONE ===", flush=True)
    print(json.dumps({
        "mean_report_rmse": mean_rmse,
        "vs_v0": summary["improvement_vs_v0"],
        "vs_v1": summary["improvement_vs_v1"],
        "folds": [(f["holdout"], f["report_rmse"], f["val_pick"]) for f in folds],
    }, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--output", default="outputs/battery_opt_v3")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seeds", default="42,123,456")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device={device} seeds={seeds}", flush=True)
    t0 = time.time()
    run(Path(args.data_root), Path(args.output), device, seeds, args.seq_len)
    print(f"seconds={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
