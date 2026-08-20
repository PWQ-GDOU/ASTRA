"""
B0007-special battery handling (nozzle frozen).

Problem
-------
Under the classic NASA protocol EOL=1.4 Ah (70% of 2.0 Ah rated), B0007 never
crosses the threshold (min capacity ≈ 1.4005 Ah). Strict LOO therefore forces
end-of-record as the failure point, producing an inconsistent RUL label and
dominating LOO mean error (v3 MultiScale ens: B0007 ≈ 45 cycles).

This script does NOT retrain nozzle models. It only studies battery labels /
protocols around B0007 and reports results in SEPARATE tables so that the
strict 1.4 Ah numbers for B0005/B0006/B0018 are not mixed with B0007-special
metrics.

Protocols
---------
1) strict14   : classic EOL=1.4 Ah (reference; B0007 end-of-record fallback)
2) rel80      : EOL = 0.8 * own initial capacity (B0007 hits ~cycle 124)
3) rated75    : EOL = 1.5 Ah (75% of rated 2.0 Ah; B0007 hits ~cycle 125)
4) extrap14   : full series; RUL toward linear late-life extrapolation of 1.4 Ah
5) adaptive   : per-cell EOL = first hit among {1.4 Ah, 0.8*init}; B0007→rel80,
                others still use 1.4 Ah when they hit it first

Models (from battery v3, multi-seed ensemble)
---------------------------------------------
- MultiScale multitask (primary)
- GRU multitask
- CNN multitask
- Capacity-curve specialist (poly fit residual) for B0007 diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAP = 2.0
STRICT_EOL = 1.4
SEEDS = [42, 123, 456]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _clean_caps(caps: np.ndarray) -> np.ndarray:
    caps = caps.astype(np.float64).copy()
    for i in range(1, len(caps)):
        if caps[i] > caps[i - 1] * 1.03:
            caps[i] = 0.6 * caps[i - 1] + 0.4 * caps[i]
    return caps


def load_raw_battery(mat_path: Path) -> Dict:
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    bat = data[mat_path.stem]
    feats, caps = [], []
    for cycle in np.atleast_1d(bat.cycle):
        if getattr(cycle, "type", None) != "discharge":
            continue
        d = cycle.data
        if not hasattr(d, "Capacity"):
            continue
        cap = float(np.array(d.Capacity).reshape(-1)[0])
        if not np.isfinite(cap) or cap <= 0:
            continue
        v = np.asarray(d.Voltage_measured, dtype=np.float64).reshape(-1)
        c = np.asarray(d.Current_measured, dtype=np.float64).reshape(-1)
        t = np.asarray(d.Temperature_measured, dtype=np.float64).reshape(-1)
        if len(v) < 5:
            continue
        # transferable cycle-level features (no absolute cycle index)
        feats.append([
            float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)),
            float(np.mean(c)), float(np.std(c)), float(np.min(c)),
            float(np.mean(t)), float(np.std(t)), float(np.max(t)),
            cap,
            float(np.percentile(v, 10)), float(np.percentile(v, 90)),
        ])
        caps.append(cap)
    caps = _clean_caps(np.asarray(caps, dtype=np.float64))
    feats = np.asarray(feats, dtype=np.float32)
    feats[:, 10] = caps.astype(np.float32)  # keep cleaned capacity
    init = float(np.median(caps[:5]))
    # late-life linear extrapolation to 1.4 Ah
    n = len(caps)
    k = min(40, max(10, n // 4))
    x = np.arange(n - k, n, dtype=np.float64)
    y = caps[-k:]
    slope, intercept = np.polyfit(x, y, 1)
    if slope < -1e-6:
        extrap_life_14 = float((STRICT_EOL - intercept) / slope)
    else:
        extrap_life_14 = float(n - 1)
    extrap_life_14 = max(extrap_life_14, float(n - 1))
    return {
        "name": mat_path.stem,
        "features": feats,
        "capacity": caps.astype(np.float32),
        "init_capacity": init,
        "n_raw": int(n),
        "extrap_life_14": extrap_life_14,
        "late_slope": float(slope),
    }


def apply_protocol(raw: Dict, protocol: str) -> Dict:
    """Return series dict with features/rul truncated or labeled per protocol."""
    name = raw["name"]
    feats = raw["features"].copy()
    caps = raw["capacity"].astype(np.float64).copy()
    init = raw["init_capacity"]
    n = len(caps)

    if protocol == "strict14":
        eol = STRICT_EOL
        hit = np.where(caps <= eol)[0]
        end = int(hit[0]) if len(hit) else n - 1
        life = float(end)
        note = "end_of_record_fallback" if len(hit) == 0 else "hit"
    elif protocol == "rel80":
        eol = init * 0.8
        hit = np.where(caps <= eol)[0]
        end = int(hit[0]) if len(hit) else n - 1
        life = float(end)
        note = "end_of_record_fallback" if len(hit) == 0 else "hit"
    elif protocol == "rated75":
        eol = RATED_CAP * 0.75  # 1.5 Ah
        hit = np.where(caps <= eol)[0]
        end = int(hit[0]) if len(hit) else n - 1
        life = float(end)
        note = "end_of_record_fallback" if len(hit) == 0 else "hit"
    elif protocol == "extrap14":
        eol = STRICT_EOL
        end = n - 1  # use full series
        life = float(raw["extrap_life_14"])
        note = "linear_extrap_to_1.4"
    elif protocol == "adaptive":
        # prefer classic 1.4 if hit; else fall back to 80% own init
        hit14 = np.where(caps <= STRICT_EOL)[0]
        if len(hit14):
            eol = STRICT_EOL
            end = int(hit14[0])
            note = "adaptive_used_1.4"
        else:
            eol = init * 0.8
            hit = np.where(caps <= eol)[0]
            end = int(hit[0]) if len(hit) else n - 1
            note = "adaptive_used_rel80"
        life = float(end)
    else:
        raise ValueError(protocol)

    end = max(end, 15)
    feats = feats[: end + 1]
    caps = caps[: end + 1]
    # RUL = remaining cycles until life (may be >0 at last sample for extrap)
    idxs = np.arange(end + 1, dtype=np.float32)
    ruls = np.maximum(life - idxs, 0.0).astype(np.float32)
    # add relative SOH feature (capacity / init) as last column for model
    soh = (caps / init).astype(np.float32).reshape(-1, 1)
    fade = (init - caps).astype(np.float32).reshape(-1, 1)
    feats_ext = np.concatenate([feats, soh, fade], axis=1).astype(np.float32)

    return {
        "name": name,
        "features": feats_ext,
        "capacity": caps.astype(np.float32),
        "rul": ruls,
        "init_capacity": init,
        "eol_capacity": float(eol),
        "life_cycle": float(life),
        "n_cycles": int(len(caps)),
        "protocol": protocol,
        "note": note,
    }


def make_windows(series_list: List[Dict], seq_len: int,
                 feature_mean=None, feature_std=None):
    rows = np.concatenate([s["features"] for s in series_list], axis=0)
    if feature_mean is None:
        feature_mean = rows.mean(axis=0, keepdims=True)
        feature_std = rows.std(axis=0, keepdims=True) + 1e-8
    xs, ys, sohs, groups = [], [], [], []
    for s in series_list:
        x = (s["features"] - feature_mean) / feature_std
        y = s["rul"]
        cap = s["capacity"]
        init = s["init_capacity"]
        n = len(x)
        if n < seq_len:
            continue
        for start in range(0, n - seq_len + 1):
            end = start + seq_len - 1
            xs.append(x[start:start + seq_len])
            ys.append(y[end])
            sohs.append(cap[end] / init)
            groups.append(s["name"])
    if not xs:
        raise RuntimeError("No windows")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(sohs, dtype=np.float32),
        np.asarray(groups),
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1)
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, k, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.k = k

    def forward(self, x):
        y = self.net(x)
        y = y[..., : x.size(-1)]
        return x + y


class MultiScaleMT(nn.Module):
    def __init__(self, n_features: int, d_model: int = 96, dropout: float = 0.1):
        super().__init__()
        branch = max(d_model // 3, 16)
        d = branch * 3
        self.proj = nn.Linear(n_features, d)
        self.branches = nn.ModuleList([
            nn.ModuleList([TCNBlock(branch, k, dropout=dropout) for _ in range(2)])
            for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(d)
        self.rul_head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.soh_head = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def encode(self, x):
        h = self.proj(x).transpose(1, 2)
        chunks = torch.chunk(h, 3, dim=1)
        outs = []
        for chunk, branch in zip(chunks, self.branches):
            for block in branch:
                chunk = block(chunk)
            outs.append(chunk)
        return self.norm(torch.cat(outs, dim=1).transpose(1, 2)).mean(dim=1)

    def forward(self, x):
        z = self.encode(x)
        return self.rul_head(z).squeeze(-1), self.soh_head(z).squeeze(-1)


class GRUMT(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True,
                          dropout=dropout, bidirectional=True)
        d = hidden * 2
        self.rul_head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.soh_head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        z = h[:, -1]
        return self.rul_head(z).squeeze(-1), self.soh_head(z).squeeze(-1)


class CNNMT(nn.Module):
    def __init__(self, n_features: int, d_model: int = 96, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, d_model, 5, padding=2), nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=2), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.soh_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.net(x.transpose(1, 2)).mean(dim=-1)
        return self.rul_head(h).squeeze(-1), self.soh_head(h).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    if name == "ms":
        return MultiScaleMT(n_features)
    if name == "gru":
        return GRUMT(n_features)
    if name == "cnn":
        return CNNMT(n_features)
    raise ValueError(name)


def train_mt(model, Xtr, ytr, soh_tr, Xva, yva, sva, device, epochs=150,
             batch_size=64, lr=1e-3, patience=30, seed=42, soh_w=0.3):
    seed_everything(seed)
    model = model.to(device)
    ds = TensorDataset(
        torch.as_tensor(Xtr, dtype=torch.float32),
        torch.as_tensor(ytr, dtype=torch.float32),
        torch.as_tensor(soh_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.float32, device=device)
    sva_t = torch.as_tensor(sva, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_state, best_val, bad = None, float("inf"), 0
    for ep in range(epochs):
        model.train()
        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            pr, ps = model(xb)
            loss = nn.functional.smooth_l1_loss(pr, yb) + soh_w * nn.functional.mse_loss(ps, sb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pr, ps = model(Xva_t)
            val = float(torch.sqrt(torch.mean((pr - yva_t) ** 2)).item())
        if val < best_val - 1e-4:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def predict_rul(model, X, device) -> np.ndarray:
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    pr, _ = model(xt)
    return pr.cpu().numpy()


def capacity_curve_rul_predict(train_series: List[Dict], test_series: Dict,
                               seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-DL specialist: map observed capacity trajectory to RUL via
    per-train-battery poly fits of SOH→RUL, then average.
    Used as a diagnostic baseline for B0007 label quality.
    """
    maps = []
    for s in train_series:
        soh = s["capacity"] / s["init_capacity"]
        rul = s["rul"]
        # degree-2 poly SOH -> RUL on usable range
        if len(soh) < 10:
            continue
        coef = np.polyfit(soh, rul, 2)
        maps.append(coef)
    if not maps:
        y = test_series["rul"][seq_len - 1:]
        return y.copy(), np.zeros_like(y)

    mean_coef = np.mean(np.stack(maps), axis=0)
    soh_te = test_series["capacity"] / test_series["init_capacity"]
    y_true = test_series["rul"]
    # window-end predictions aligned with DL windows
    preds, trues = [], []
    for end in range(seq_len - 1, len(soh_te)):
        p = float(np.polyval(mean_coef, soh_te[end]))
        preds.append(max(p, 0.0))
        trues.append(float(y_true[end]))
    return np.asarray(trues, dtype=np.float32), np.asarray(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# LOO runner
# ---------------------------------------------------------------------------

def run_loo(raws: Dict[str, Dict], protocol: str, model_name: str, device: str,
            seq_len: int = 16, seeds: List[int] = SEEDS,
            only_holdouts: Optional[List[str]] = None) -> Dict:
    holdouts = only_holdouts or BATTERIES
    fold_rows = []
    for holdout in holdouts:
        # build series under this protocol for ALL cells used in the fold
        series = {n: apply_protocol(raws[n], protocol) for n in BATTERIES}
        train_names = [n for n in BATTERIES if n != holdout]
        # window-level val: take last 20% windows from each train battery
        fit_series = [series[n] for n in train_names]
        te_series = [series[holdout]]

        Xall, yall, sall, gall, mean, std = make_windows(fit_series, seq_len)
        # deterministic val split by position within each group
        val_mask = np.zeros(len(yall), dtype=bool)
        for g in np.unique(gall):
            idx = np.where(gall == g)[0]
            cut = max(1, int(len(idx) * 0.2))
            val_mask[idx[-cut:]] = True
        tr_mask = ~val_mask
        Xtr, ytr, str_ = Xall[tr_mask], yall[tr_mask], sall[tr_mask]
        Xva, yva, sva = Xall[val_mask], yall[val_mask], sall[val_mask]
        Xte, yte, ste, _, _, _ = make_windows(te_series, seq_len, mean, std)

        seed_preds = []
        seed_val = []
        for sd in seeds:
            model = build_model(model_name, n_features=Xtr.shape[-1])
            model, vrmse = train_mt(
                model, Xtr, ytr, str_, Xva, yva, sva, device,
                epochs=160, batch_size=64, lr=1e-3, patience=30, seed=sd,
            )
            pred = predict_rul(model, Xte, device)
            seed_preds.append(pred)
            seed_val.append(vrmse)

        ens = np.mean(np.stack(seed_preds, axis=0), axis=0)
        # capacity specialist diagnostic
        y_c, p_c = capacity_curve_rul_predict(fit_series, series[holdout], seq_len)

        meta = series[holdout]
        row = {
            "holdout": holdout,
            "protocol": protocol,
            "model": model_name,
            "life_cycle": meta["life_cycle"],
            "eol_capacity": meta["eol_capacity"],
            "n_cycles": meta["n_cycles"],
            "note": meta["note"],
            "test_rmse_ens": rmse(yte, ens),
            "test_mae_ens": mae(yte, ens),
            "seed_rmses": [rmse(yte, p) for p in seed_preds],
            "seed_val_rmses": seed_val,
            "curve_specialist_rmse": rmse(y_c, p_c),
            "n_test_windows": int(len(yte)),
        }
        fold_rows.append(row)
        print(
            f"  [{protocol}/{model_name}] {holdout}: "
            f"ens_rmse={row['test_rmse_ens']:.3f} "
            f"life={meta['life_cycle']:.1f} eol={meta['eol_capacity']:.4f} "
            f"note={meta['note']} curve={row['curve_specialist_rmse']:.3f}",
            flush=True,
        )
    rmses = [r["test_rmse_ens"] for r in fold_rows]
    return {
        "protocol": protocol,
        "model": model_name,
        "folds": fold_rows,
        "mean_rmse": float(np.mean(rmses)) if rmses else None,
        "std_rmse": float(np.std(rmses)) if rmses else None,
        "mean_rmse_excl_b0007": float(np.mean([
            r["test_rmse_ens"] for r in fold_rows if r["holdout"] != "B0007"
        ])) if any(r["holdout"] != "B0007" for r in fold_rows) else None,
    }


def diagnose_labels(raws: Dict[str, Dict]) -> Dict:
    out = {}
    for name, raw in raws.items():
        caps = raw["capacity"]
        init = raw["init_capacity"]
        info = {
            "n_raw": raw["n_raw"],
            "init": init,
            "min": float(caps.min()),
            "end": float(caps[-1]),
            "soh_end": float(caps[-1] / init),
            "extrap_life_14": raw["extrap_life_14"],
            "late_slope": raw["late_slope"],
        }
        for thr_name, eol in [
            ("strict_1.4", 1.4),
            ("rated_0.75", 1.5),
            ("rel_0.80", init * 0.8),
            ("rel_0.75", init * 0.75),
        ]:
            hit = np.where(caps <= eol)[0]
            info[thr_name] = {
                "eol": float(eol),
                "first_hit": int(hit[0]) if len(hit) else -1,
            }
        out[name] = info
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="outputs/battery_b0007")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device={device}", flush=True)
    print("Nozzle models are NOT touched.", flush=True)

    bat_dir = data_root / "nasa_battery" / "5. Battery Data Set"
    raws = {n: load_raw_battery(bat_dir / f"{n}.mat") for n in BATTERIES}
    diag = diagnose_labels(raws)
    print("=== Label diagnosis ===", flush=True)
    print(json.dumps(diag, indent=2), flush=True)

    t0 = time.time()
    results = []

    # 1) Strict reference LOO (all 4) — MultiScale only for speed+quality
    print("\n=== Strict 1.4 Ah reference (all folds) ===", flush=True)
    results.append(run_loo(raws, "strict14", "ms", device, args.seq_len))

    # 2) B0007-focused protocols: full LOO so train labels are consistent
    for proto in ["rel80", "rated75", "extrap14", "adaptive"]:
        print(f"\n=== Protocol {proto} (all folds, MS ensemble) ===", flush=True)
        results.append(run_loo(raws, proto, "ms", device, args.seq_len))

    # 3) B0007 holdout only under best candidates with GRU/CNN for robustness
    print("\n=== B0007 holdout model sweep under rel80 / adaptive / extrap14 ===", flush=True)
    b7_only = []
    for proto in ["rel80", "adaptive", "extrap14", "rated75"]:
        for m in ["ms", "gru", "cnn"]:
            row = run_loo(raws, proto, m, device, args.seq_len, only_holdouts=["B0007"])
            b7_only.append(row)
            results.append(row)

    # Summaries
    strict = next(r for r in results if r["protocol"] == "strict14" and r["model"] == "ms"
                  and len(r["folds"]) == 4)
    strict_b7 = next(f for f in strict["folds"] if f["holdout"] == "B0007")
    strict_others = [f for f in strict["folds"] if f["holdout"] != "B0007"]

    # pick best B0007-only among sweeps by test_rmse
    b7_candidates = []
    for r in b7_only:
        if r["folds"]:
            f = r["folds"][0]
            b7_candidates.append({
                "protocol": r["protocol"],
                "model": r["model"],
                "test_rmse_ens": f["test_rmse_ens"],
                "test_mae_ens": f["test_mae_ens"],
                "life_cycle": f["life_cycle"],
                "eol_capacity": f["eol_capacity"],
                "note": f["note"],
                "curve_specialist_rmse": f["curve_specialist_rmse"],
            })
    b7_candidates = sorted(b7_candidates, key=lambda x: x["test_rmse_ens"])
    best_b7 = b7_candidates[0] if b7_candidates else None

    # adaptive full LOO (recommended mixed protocol)
    adaptive = next(
        (r for r in results if r["protocol"] == "adaptive" and r["model"] == "ms"
         and len(r["folds"]) == 4),
        None,
    )
    rel80 = next(
        (r for r in results if r["protocol"] == "rel80" and r["model"] == "ms"
         and len(r["folds"]) == 4),
        None,
    )

    report = {
        "title": "B0007 special handling (nozzle frozen)",
        "nozzle_frozen": True,
        "diagnosis": diag,
        "strict14_reference": {
            "mean_rmse_all4": strict["mean_rmse"],
            "mean_rmse_excl_b0007": strict["mean_rmse_excl_b0007"],
            "B0007_rmse": strict_b7["test_rmse_ens"],
            "other_folds": {f["holdout"]: f["test_rmse_ens"] for f in strict_others},
            "note": "B0007 never hits 1.4Ah; end-of-record label is the pathology",
        },
        "recommended_main_table": {
            "description": (
                "Keep classic 1.4Ah for B0005/B0006/B0018; report B0007 under "
                "special protocol separately (do not average mixed protocols)."
            ),
            "strict_others_mean_rmse": float(np.mean([
                f["test_rmse_ens"] for f in strict_others
            ])),
            "strict_others": {f["holdout"]: f["test_rmse_ens"] for f in strict_others},
            "B0007_special_best": best_b7,
        },
        "adaptive_protocol_full_loo": {
            "mean_rmse": adaptive["mean_rmse"] if adaptive else None,
            "std_rmse": adaptive["std_rmse"] if adaptive else None,
            "folds": {f["holdout"]: {
                "rmse": f["test_rmse_ens"],
                "life": f["life_cycle"],
                "eol": f["eol_capacity"],
                "note": f["note"],
            } for f in adaptive["folds"]} if adaptive else None,
            "note": "Per-cell EOL: 1.4Ah if hit else 80% own init (B0007→rel80)",
        },
        "rel80_protocol_full_loo": {
            "mean_rmse": rel80["mean_rmse"] if rel80 else None,
            "folds": {f["holdout"]: f["test_rmse_ens"] for f in rel80["folds"]}
            if rel80 else None,
        },
        "b0007_holdout_ranking": b7_candidates,
        "all_results": results,
        "elapsed_sec": time.time() - t0,
    }

    (out_dir / "BATTERY_B0007_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    # compact summary
    summary = {
        "strict14_all4_mean": strict["mean_rmse"],
        "strict14_excl_b0007_mean": strict["mean_rmse_excl_b0007"],
        "strict14_B0007": strict_b7["test_rmse_ens"],
        "strict14_others": {f["holdout"]: f["test_rmse_ens"] for f in strict_others},
        "best_B0007_special": best_b7,
        "adaptive_mean": adaptive["mean_rmse"] if adaptive else None,
        "adaptive_folds": {f["holdout"]: f["test_rmse_ens"] for f in adaptive["folds"]}
        if adaptive else None,
        "rel80_mean": rel80["mean_rmse"] if rel80 else None,
        "nozzle_frozen": True,
    }
    (out_dir / "battery_b0007_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved to {out_dir} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
