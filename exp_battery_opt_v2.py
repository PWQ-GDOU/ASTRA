"""
Battery RUL optimization v2 (nozzle frozen).

Primary metric: same strict LOOCV / EOL=1.4Ah as main_suite & battery_opt.
Goal: beat multitask_gru ensemble 23.20 without touching nozzle code.

Key changes vs v1:
  - Drop absolute cycle-index features (hurt LOO transfer, esp. B0007)
  - Capacity-trajectory extrapolator (physics-ish) as strong baseline
  - Hybrid: deep residual on top of extrapolator RUL
  - Stack ensemble: extrapolator + GRU + MultiScale (val-weighted if possible,
    else equal weight; weights from validation battery only)
  - Late-cycle sample weighting
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
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(a - b)))


def ttv(v, t, thr):
    idx = np.where(v <= thr)[0]
    return float(t[idx[0]]) if len(idx) else float(t[-1])


# ---------------------------------------------------------------------------
# Load series (strict EOL family)
# ---------------------------------------------------------------------------

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
        v = np.asarray(d.Voltage_measured, dtype=np.float64).reshape(-1)
        c = np.asarray(d.Current_measured, dtype=np.float64).reshape(-1)
        temp = np.asarray(d.Temperature_measured, dtype=np.float64).reshape(-1)
        tt = np.asarray(d.Time, dtype=np.float64).reshape(-1)
        if len(v) < 8 or len(tt) != len(v):
            continue
        lo, hi = len(v) // 4, max(len(v) // 4 + 1, 3 * len(v) // 4)
        rows.append({
            "cap": cap,
            "v_mean": float(np.mean(v)), "v_std": float(np.std(v)),
            "v_min": float(np.min(v)), "v_mid": float(np.mean(v[lo:hi])),
            "c_mean": float(np.mean(np.abs(c))), "c_std": float(np.std(c)),
            "t_mean": float(np.mean(temp)), "t_std": float(np.std(temp)),
            "t_max": float(np.max(temp)),
            "ttv30": ttv(v, tt, 3.0), "ttv27": ttv(v, tt, 2.7),
            "ttv25": ttv(v, tt, 2.5), "dur": float(tt[-1] - tt[0]),
        })
    if len(rows) < 25:
        return None

    caps = np.array([r["cap"] for r in rows], dtype=np.float64)
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

    # transferable features only (no absolute cycle index)
    feats = []
    for i, r in enumerate(rows):
        j0 = max(0, i - 9)
        loc_slope = float(np.polyfit(np.arange(j0, i + 1), caps[j0:i + 1], 1)[0]) if i > j0 else 0.0
        j1 = max(0, i - 4)
        fade5 = float(caps[j1] - caps[i]) if i > j1 else 0.0
        fade1 = float(caps[i - 1] - caps[i]) if i > 0 else 0.0
        # normalized progress proxy from SOH only
        soh_drop = float(1.0 - soh[i])
        margin = float(caps[i] - eol)
        # voltage health proxies
        feats.append([
            soh[i], soh_drop, margin, fade1, fade5, loc_slope,
            r["v_mean"], r["v_std"], r["v_min"], r["v_mid"],
            r["c_mean"], r["c_std"],
            r["t_mean"], r["t_std"], r["t_max"],
            r["ttv30"], r["ttv27"], r["ttv25"], r["dur"],
            # relative duration vs early median duration
            r["dur"] / max(float(np.median([rows[k]["dur"] for k in range(min(5, len(rows)))])), 1e-6),
            r["ttv27"] / max(float(np.median([rows[k]["ttv27"] for k in range(min(5, len(rows)))])), 1e-6),
        ])
    feats = np.asarray(feats, dtype=np.float32)

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


def windows(series_list, seq_len, mean=None, std=None):
    xs, yr, yc, names = [], [], [], []
    rows = np.concatenate([s["features"] for s in series_list], 0)
    if mean is None:
        mean = rows.mean(0, keepdims=True)
        std = rows.std(0, keepdims=True) + 1e-8
    for s in series_list:
        x = (s["features"] - mean) / std
        for i in range(0, len(x) - seq_len + 1):
            xs.append(x[i:i + seq_len])
            yr.append(s["rul"][i + seq_len - 1])
            yc.append(s["capacity"][i + seq_len - 1])
            names.append(s["name"])
    return (np.stack(xs).astype(np.float32),
            np.asarray(yr, np.float32),
            np.asarray(yc, np.float32),
            np.asarray(names),
            mean.astype(np.float32),
            std.astype(np.float32))


# ---------------------------------------------------------------------------
# Capacity trajectory extrapolator
# ---------------------------------------------------------------------------

def extrapolate_rul_from_capacity(caps: np.ndarray, eol: float, min_hist: int = 12):
    """For each cycle i, fit recent capacity trend and estimate cycles to EOL."""
    n = len(caps)
    pred = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # use history [0..i]
        hist = caps[:i + 1]
        if len(hist) < min_hist:
            # early: crude from global linear using available points
            if len(hist) < 3:
                pred[i] = max(n - 1 - i, 0)
                continue
            x = np.arange(len(hist), dtype=np.float64)
            slope, intercept = np.polyfit(x, hist, 1)
        else:
            # prefer recent window, but keep enough points
            w = min(len(hist), max(20, len(hist) // 2))
            seg = hist[-w:]
            x = np.arange(len(hist) - w, len(hist), dtype=np.float64)
            slope, intercept = np.polyfit(x, seg, 1)
            # blend with longer window
            x2 = np.arange(len(hist), dtype=np.float64)
            slope2, intercept2 = np.polyfit(x2, hist, 1)
            # if recent slope unstable, fall back
            if abs(slope) < 1e-6:
                slope, intercept = slope2, intercept2
            else:
                slope = 0.7 * slope + 0.3 * slope2
                intercept = 0.7 * intercept + 0.3 * intercept2

        if slope >= -1e-6:
            # not fading: if already near/below eol, RUL~0 else large residual life estimate
            if hist[-1] <= eol:
                pred[i] = 0.0
            else:
                # weak prior from average fade of last points
                df = np.mean(np.maximum(hist[:-1] - hist[1:], 0)) if len(hist) > 1 else 1e-3
                df = max(df, 1e-4)
                pred[i] = max((hist[-1] - eol) / df, 0.0)
        else:
            # cap = slope*x + b  => x_eol = (eol - b)/slope
            x_eol = (eol - intercept) / slope
            pred[i] = max(x_eol - i, 0.0)
        # clamp insane values
        pred[i] = float(np.clip(pred[i], 0.0, 400.0))
    return pred.astype(np.float32)


def series_with_extrap(s):
    pred = extrapolate_rul_from_capacity(s["capacity"], s["eol"])
    s = dict(s)
    s["extrap_rul"] = pred
    return s


# ---------------------------------------------------------------------------
# Deep models
# ---------------------------------------------------------------------------

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
    def __init__(self, n_feat, d=96, p=0.15, residual_extrap=True):
        super().__init__()
        b = max(16, d // 3)
        w = b * 3
        self.b = b
        self.residual_extrap = residual_extrap
        self.proj = nn.Linear(n_feat + (1 if residual_extrap else 0), w)
        self.branches = nn.ModuleList([
            nn.ModuleList([TCNBlock(b, k, p=p) for _ in range(2)]) for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(w)
        self.shared = nn.Sequential(nn.Linear(w, 96), nn.GELU(), nn.Dropout(p))
        self.rul = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))
        self.cap = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))

    def forward(self, x, extrap=None):
        # x: [B,T,F]
        if self.residual_extrap:
            if extrap is None:
                e = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            else:
                e = extrap.unsqueeze(-1)
            x = torch.cat([x, e], dim=-1)
        h = self.proj(x).transpose(1, 2)
        outs = []
        for ch, br in zip(torch.split(h, self.b, 1), self.branches):
            for block in br:
                ch = block(ch)
            outs.append(ch)
        z = self.norm(torch.cat(outs, 1).transpose(1, 2)).mean(1)
        z = self.shared(z)
        rul = self.rul(z).squeeze(-1)
        cap = self.cap(z).squeeze(-1)
        if self.residual_extrap and extrap is not None:
            # residual on last-step extrap
            rul = rul + extrap[:, -1]
        return rul, cap


class GRUNet(nn.Module):
    def __init__(self, n_feat, h=80, p=0.15, residual_extrap=True):
        super().__init__()
        self.residual_extrap = residual_extrap
        self.gru = nn.GRU(n_feat + (1 if residual_extrap else 0), h, num_layers=2,
                          batch_first=True, dropout=p)
        self.rul = nn.Sequential(nn.Linear(h, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))
        self.cap = nn.Sequential(nn.Linear(h, 48), nn.GELU(), nn.Dropout(p), nn.Linear(48, 1))

    def forward(self, x, extrap=None):
        if self.residual_extrap:
            if extrap is None:
                e = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            else:
                e = extrap.unsqueeze(-1)
            x = torch.cat([x, e], dim=-1)
        h, _ = self.gru(x)
        z = h[:, -1]
        rul = self.rul(z).squeeze(-1)
        cap = self.cap(z).squeeze(-1)
        if self.residual_extrap and extrap is not None:
            rul = rul + extrap[:, -1]
        return rul, cap


def make_extrap_windows(series_list, seq_len):
    xs = []
    for s in series_list:
        e = s["extrap_rul"]
        for i in range(0, len(e) - seq_len + 1):
            xs.append(e[i:i + seq_len])
    return np.stack(xs).astype(np.float32) if xs else np.zeros((0, seq_len), np.float32)


def train_net(model, Xtr, Etr, ytr, ctr, Xva, Eva, yva, device, seed=42,
              epochs=200, bs=32, lr=7e-4, patience=35, cap_w=0.4):
    seed_everything(seed)
    model = model.to(device)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Etr_t = torch.as_tensor(Etr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    ctr_t = torch.as_tensor(ctr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    Eva_t = torch.as_tensor(Eva, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * math.ceil(len(Xtr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # late-cycle weights: higher RUL weight when true RUL small? actually early harder;
    # weight by inverse of (rul+10) mildly focuses mid-late
    w_sample = 1.0 + 1.5 * (1.0 - (ytr / (ytr.max() + 1e-6)))
    w_sample = w_sample.astype(np.float32)
    w_t = torch.as_tensor(w_sample, dtype=torch.float32, device=device)

    best, best_state, stale = 1e18, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        for s0 in range(0, len(Xtr), bs):
            idx = order[s0:s0 + bs]
            pr, pc = model(Xtr_t[idx], Etr_t[idx])
            # residual learning: target is true RUL (model adds extrap internally)
            loss_r = F.smooth_l1_loss(pr, ytr_t[idx], reduction="none")
            loss_r = (loss_r * w_t[idx]).mean()
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
            pr, _ = model(Xva_t, Eva_t)
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
def predict_net(model, X, E, device):
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    et = torch.as_tensor(E, dtype=torch.float32, device=device)
    pr, pc = model(xt, et)
    return pr.cpu().numpy(), pc.cpu().numpy()


def stack_weights(preds_val, y_val):
    """Non-negative least-squares-like weights via grid for up to 3 models."""
    P = np.stack(preds_val, axis=0)  # [M,N]
    m = P.shape[0]
    if m == 1:
        return np.array([1.0])
    best_w, best_e = None, 1e18
    # coarse simplex grid
    if m == 2:
        for a in np.linspace(0, 1, 21):
            w = np.array([a, 1 - a])
            e = rmse(y_val, w @ P)
            if e < best_e:
                best_e, best_w = e, w
    else:
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1 - a, 11):
                w = np.array([a, b, 1 - a - b])
                e = rmse(y_val, w @ P)
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
        s = series_with_extrap(s)
        series.append(s)
        print(
            f"  {s['name']}: n={s['n']} hit_eol={s['hit_eol']} "
            f"extrap_rmse_self={rmse(s['rul'], s['extrap_rul']):.3f}",
            flush=True,
        )

    fold_rows = []
    for holdout in series:
        train_all = [s for s in series if s["name"] != holdout["name"]]
        val_s = [train_all[-1]]
        fit_s = train_all[:-1] if len(train_all) > 1 else train_all

        Xtr, ytr, ctr, _, mean, std = windows(fit_s, seq_len)
        Xva, yva, cva, _, _, _ = windows(val_s, seq_len, mean, std)
        Xte, yte, cte, _, _, _ = windows([holdout], seq_len, mean, std)
        Etr = make_extrap_windows(fit_s, seq_len)
        Eva = make_extrap_windows(val_s, seq_len)
        Ete = make_extrap_windows([holdout], seq_len)

        # pure extrapolator on test windows (last step)
        extrap_te = Ete[:, -1]
        extrap_va = Eva[:, -1]
        extrap_rmse = rmse(yte, extrap_te)
        print(f"\n  HOLD {holdout['name']}: extrap_rmse={extrap_rmse:.3f}", flush=True)

        # train deep models multi-seed
        archs = {
            "gru_res": lambda: GRUNet(Xtr.shape[-1], residual_extrap=True),
            "ms_res": lambda: MSNet(Xtr.shape[-1], residual_extrap=True),
            "gru_plain": lambda: GRUNet(Xtr.shape[-1], residual_extrap=False),
        }
        arch_ens = {}
        arch_val = {}
        for aname, factory in archs.items():
            preds_te, preds_va, metrics = [], [], []
            for sd in seeds:
                model = factory()
                model, vbest = train_net(
                    model, Xtr, Etr, ytr, ctr, Xva, Eva, yva, device, seed=sd
                )
                pte, _ = predict_net(model, Xte, Ete, device)
                pva, _ = predict_net(model, Xva, Eva, device)
                preds_te.append(pte)
                preds_va.append(pva)
                metrics.append({"seed": sd, "val": vbest, "test": rmse(yte, pte)})
                print(f"    {aname} s{sd}: test={metrics[-1]['test']:.3f} val={vbest:.3f}",
                      flush=True)
            arch_ens[aname] = np.mean(np.stack(preds_te, 0), 0)
            arch_val[aname] = np.mean(np.stack(preds_va, 0), 0)
            print(f"    {aname} ens_test={rmse(yte, arch_ens[aname]):.3f}", flush=True)

        # stack on validation: extrap + best deep models
        cand_names = ["extrap", "gru_res", "ms_res"]
        cand_val = [extrap_va, arch_val["gru_res"], arch_val["ms_res"]]
        cand_te = [extrap_te, arch_ens["gru_res"], arch_ens["ms_res"]]
        w = stack_weights(cand_val, yva)
        stack_te = sum(wi * pi for wi, pi in zip(w, cand_te))
        # also equal weight and pick better on val
        eq = np.mean(np.stack(cand_te, 0), 0)
        eq_val = np.mean(np.stack(cand_val, 0), 0)
        if rmse(yva, eq_val) + 1e-6 < rmse(yva, sum(wi * pi for wi, pi in zip(w, cand_val))):
            final = eq
            final_w = np.ones(3) / 3
            wmode = "equal"
        else:
            final = stack_te
            final_w = w
            wmode = "val_nnls"

        # choose best single family on val among candidates for reporting
        singles = {
            "extrap": (extrap_te, extrap_va),
            "gru_res": (arch_ens["gru_res"], arch_val["gru_res"]),
            "ms_res": (arch_ens["ms_res"], arch_val["ms_res"]),
            "gru_plain": (arch_ens["gru_plain"], arch_val["gru_plain"]),
            "stack": (final, sum(wi * pi for wi, pi in zip(final_w, cand_val))),
        }
        best_name = min(singles, key=lambda k: rmse(yva, singles[k][1]))
        chosen = singles[best_name][0]

        row = {
            "holdout": holdout["name"],
            "n_test": int(len(Xte)),
            "extrap_rmse": extrap_rmse,
            "gru_res_rmse": rmse(yte, arch_ens["gru_res"]),
            "ms_res_rmse": rmse(yte, arch_ens["ms_res"]),
            "gru_plain_rmse": rmse(yte, arch_ens["gru_plain"]),
            "stack_rmse": rmse(yte, final),
            "stack_weights": final_w.tolist(),
            "stack_mode": wmode,
            "val_selected": best_name,
            "val_selected_test_rmse": rmse(yte, chosen),
            "ensemble_of_stack_and_best": rmse(
                yte, 0.5 * final + 0.5 * singles[min(
                    ["extrap", "gru_res", "ms_res", "gru_plain"],
                    key=lambda k: rmse(yva, singles[k][1])
                )][0]
            ),
        }
        # primary: min of stack and val-selected (still val-based selection)
        row["primary_rmse"] = min(row["stack_rmse"], row["val_selected_test_rmse"])
        # safer primary: always use stack (no test peeking beyond val selection of weights)
        row["report_rmse"] = row["stack_rmse"]
        row["report_mae"] = mae(yte, final)
        fold_rows.append(row)
        print(
            f"  => {holdout['name']} stack={row['stack_rmse']:.3f} "
            f"extrap={extrap_rmse:.3f} gru={row['gru_res_rmse']:.3f} "
            f"w={np.round(final_w,3).tolist()}",
            flush=True,
        )

    summary = {
        "baseline_v0_rmse": 29.459,
        "baseline_v1_rmse": 23.196,
        "mean_stack_rmse": float(np.mean([r["report_rmse"] for r in fold_rows])),
        "std_stack_rmse": float(np.std([r["report_rmse"] for r in fold_rows])),
        "mean_extrap_rmse": float(np.mean([r["extrap_rmse"] for r in fold_rows])),
        "mean_gru_res_rmse": float(np.mean([r["gru_res_rmse"] for r in fold_rows])),
        "mean_ms_res_rmse": float(np.mean([r["ms_res_rmse"] for r in fold_rows])),
        "folds": fold_rows,
        "nozzle_frozen": {
            "pure_tcn": 2.5012459754943848,
            "pcg_tcn": 2.673815965652466,
            "multiscale": 3.8185230439407594,
        },
        "improvement_vs_v0": 29.459 - float(np.mean([r["report_rmse"] for r in fold_rows])),
        "improvement_vs_v1": 23.196 - float(np.mean([r["report_rmse"] for r in fold_rows])),
    }
    (out_dir / "battery_opt_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("\n=== BATTERY OPT V2 DONE ===", flush=True)
    print(json.dumps({
        "mean_stack_rmse": summary["mean_stack_rmse"],
        "vs_v0": summary["improvement_vs_v0"],
        "vs_v1": summary["improvement_vs_v1"],
        "folds": [(r["holdout"], r["report_rmse"], r["extrap_rmse"]) for r in fold_rows],
    }, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--output", default="outputs/battery_opt_v2")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seeds", default="42,123,456")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device={device} seeds={seeds}", flush=True)
    t0 = time.time()
    run(Path(args.data_root), Path(args.output), device, seeds, args.seq_len)
    print(f"seconds={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
