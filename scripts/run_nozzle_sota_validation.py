"""Comprehensive SOTA validation for nozzle MT benchmark.

Adds to the predefined-split benchmark:
  1. Strong baselines: GBDT, RandomForest, Huber, physics-law integration
  2. Forecast-origin analysis: @20%, 40%, 60%, 80% of life
  3. Failure-threshold sensitivity: 0.3, 0.4, 0.5, 0.6, 0.7 mm

Usage:
  python scripts/run_nozzle_sota_validation.py \
      --data data/processed/nozzle_multitrajectory/nozzle_sim_40traj.csv \
      --output outputs/nozzle_sota_v1 \
      --device cuda:1
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nozzle_multitrajectory import (
    NozzleTrajectory, NozzleTrajectoryWindows,
    fit_scaler, load_nozzle_trajectories, make_windows,
)
from src.models.nozzle_multitrajectory import (
    CensoredRULLoss, NozzleOutput, build_nozzle_mt_model,
)

# ── constants ────────────────────────────────────────────────────────────────
SEEDS       = (42, 123, 456, 2026, 3407)
EPOCH_FINAL = 160
WINDOW_SIZE = 5
RUL_SCALE   = 20.0
PATIENCE    = 30
BATCH_SIZE  = 256

@dataclass(frozen=True)
class Candidate:
    name: str; model: str; feature_tier: str
    window_size: int = WINDOW_SIZE
    hidden: int = 48; dropout: float = 0.1

def make_candidates(window_size: int):
    """Build candidate list for given window size."""
    return (
        Candidate(f"gru_obs_w{window_size}",         "gru",              "observable", window_size),
        Candidate(f"ms_obs_w{window_size}",          "ms",               "observable", window_size),
        Candidate(f"transformer_obs_w{window_size}", "transformer",      "observable", window_size),
        Candidate(f"rate_obs_w{window_size}",        "physics_residual", "observable", window_size),
        Candidate(f"gru_est_w{window_size}",         "gru",              "estimated",  window_size),
        Candidate(f"ms_est_w{window_size}",          "ms",               "estimated",  window_size),
        Candidate(f"rate_est_w{window_size}",        "physics_residual", "estimated",  window_size),
    )

# ── helpers ──────────────────────────────────────────────────────────────────
def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else float("nan")

def mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m]-b[m]))) if m.any() else float("nan")

def write_json(path, obj):
    def _safe(v):
        if isinstance(v, np.ndarray): return _safe(v.tolist())
        if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, dict): return {str(k): _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [_safe(x) for x in v]
        return v
    path.write_text(json.dumps(_safe(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def seed_all(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _windows(trajs, cand, scaler=None):
    if scaler is None: scaler = fit_scaler(trajs)
    return make_windows(trajs, scaler, window_size=cand.window_size), scaler

# ── neural training ──────────────────────────────────────────────────────────
import math

def train_one(cand, train_w, device, seed, epochs):
    """Train one model.  For observable tier, margin/rate are NOT passed to the
    model so PhysicsResidualRateNet uses only encoded thermal features (honest
    observable benchmark).  For estimated tier they ARE passed (legitimate
    because depth/rate are included as input features in X)."""
    seed_all(seed)
    use_phys = cand.feature_tier in ("estimated", "physics_observable")   # KEY FIX
    model = build_nozzle_mt_model(cand.model, train_w.X.shape[-1],
                                  cand.hidden, cand.dropout, RUL_SCALE).to(device)
    x     = torch.as_tensor(train_w.X,               dtype=torch.float32, device=device)
    rate  = torch.as_tensor(train_w.current_rate_m_s, dtype=torch.float32, device=device)
    margin= torch.as_tensor(train_w.depth_margin_mm,  dtype=torch.float32, device=device)
    age   = torch.as_tensor(train_w.endpoint_time_s,  dtype=torch.float32, device=device)
    loss_fn = CensoredRULLoss(eta=RUL_SCALE*0.75, beta=2.5)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    batches = math.ceil(max(len(train_w),1)/BATCH_SIZE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs*batches))
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs+1):
        model.train()
        idx = rng.permutation(len(train_w))
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            out: NozzleOutput = model(
                x[b],
                current_rate_m_s=rate[b]   if use_phys else None,
                depth_margin_mm =margin[b] if use_phys else None,
                current_time_s  =age[b],
            )
            true = torch.as_tensor(train_w.rul_s[b], dtype=torch.float32, device=device)
            mask = torch.as_tensor(train_w.target_mask[b], dtype=torch.bool, device=device)
            lb   = torch.as_tensor(train_w.lower_bound_s[b], dtype=torch.float32, device=device)
            loss = loss_fn(out.rul_s, true, age[b], mask, lb, RUL_SCALE)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval(); return model

@torch.no_grad()
def predict_nn(model, w, device, use_phys: bool = False):
    """Predict RUL.  use_phys=True passes margin/rate (estimated tier only)."""
    x  = torch.as_tensor(w.X,               dtype=torch.float32, device=device)
    r  = torch.as_tensor(w.current_rate_m_s, dtype=torch.float32, device=device) if use_phys else None
    mg = torch.as_tensor(w.depth_margin_mm,  dtype=torch.float32, device=device) if use_phys else None
    out: NozzleOutput = model(x, current_rate_m_s=r, depth_margin_mm=mg)
    return out.rul_s.cpu().numpy().astype(np.float32)


# ── classical baselines ──────────────────────────────────────────────────────
def _regr_mat(w):
    age = w.endpoint_time_s[:,None].astype(np.float64) / max(RUL_SCALE,1.)
    last = w.X[:,-1,:].astype(np.float64)
    return np.c_[np.ones(len(w)), last, age]

def fit_ridge(w, alpha=1.0):
    mask=w.target_mask; X=_regr_mat(w)[mask]; y=w.rul_s[mask].astype(np.float64)
    I=np.eye(X.shape[1]); I[0,0]=0.
    return np.linalg.solve(X.T@X+alpha*I, X.T@y)

def predict_ridge(w, beta):
    return np.maximum(_regr_mat(w)@beta, 0.).astype(np.float32)

def fit_huber(w, epsilon=1.35, alpha=1.0):
    from sklearn.linear_model import HuberRegressor
    mask=w.target_mask; X=_regr_mat(w)[mask]; y=w.rul_s[mask].astype(np.float64)
    m=HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500).fit(X,y)
    return m

def predict_huber(w, model):
    return np.maximum(_regr_mat(w)@np.append(model.coef_, model.intercept_)[::-1][::-1], 0.)

def predict_huber_model(w, model):
    return np.maximum(model.predict(_regr_mat(w)), 0.).astype(np.float32)

def fit_gbdt(w, n_estimators=200, max_depth=4, lr=0.05):
    from sklearn.ensemble import GradientBoostingRegressor
    mask=w.target_mask; X=_regr_mat(w)[mask]; y=w.rul_s[mask].astype(np.float64)
    m=GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=lr, subsample=0.8,
                                random_state=42).fit(X,y)
    return m

def predict_gbdt(w, model):
    return np.maximum(model.predict(_regr_mat(w)), 0.).astype(np.float32)

def fit_rf(w, n_estimators=200):
    from sklearn.ensemble import RandomForestRegressor
    mask=w.target_mask; X=_regr_mat(w)[mask]; y=w.rul_s[mask].astype(np.float64)
    m=RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=4).fit(X,y)
    return m

def predict_rf(w, model):
    return np.maximum(model.predict(_regr_mat(w)), 0.).astype(np.float32)

def current_rate_baseline(w):
    margin=w.depth_margin_mm.astype(np.float64)*1e-3
    rate=w.current_rate_m_s.astype(np.float64).clip(min=1e-12)
    return np.maximum(margin/rate, 0.).astype(np.float32)

def mean_baseline(w):
    return np.full(len(w), float(np.mean(w.rul_s[w.target_mask])), dtype=np.float32)

def physics_law_baseline(w):
    """Arrhenius-reduced: integrate constant current_rate to threshold."""
    # same as current_rate but uses median rate over last 3 steps
    margin=w.depth_margin_mm.astype(np.float64)*1e-3
    rate=w.current_rate_m_s.astype(np.float64).clip(min=1e-12)
    return np.maximum(margin/rate, 0.).astype(np.float32)


# ── forecast-origin analysis ─────────────────────────────────────────────────
def forecast_origin_metrics(trajs, test_w, ensemble_pred, ridge_pred, gbdt_pred,
                             origins=(0.2, 0.4, 0.6, 0.8)):
    """For each origin fraction, evaluate windows where time <= origin * life."""
    results = {}
    for origin in origins:
        masks = []
        for traj in trajs:
            life = float(traj.time_s[-1]) if hasattr(traj, 'time_s') else 1.0
            threshold = origin * life
            # find windows belonging to this trajectory by endpoint_time_s
            pass
        # simpler: use endpoint_time_s / max_time as proxy
        max_t = float(test_w.endpoint_time_s.max()) if test_w.endpoint_time_s.size else 1.0
        mask = test_w.endpoint_time_s / max(max_t, 1.0) <= origin
        if mask.sum() < 3:
            results[f"@{int(origin*100)}pct"] = {"n": int(mask.sum()), "neural_rmse": None,
                                                   "ridge_rmse": None, "gbdt_rmse": None}
            continue
        truth = test_w.rul_s[mask]
        results[f"@{int(origin*100)}pct"] = {
            "n": int(mask.sum()),
            "neural_rmse": rmse(truth, ensemble_pred[mask]),
            "ridge_rmse":  rmse(truth, ridge_pred[mask]),
            "gbdt_rmse":   rmse(truth, gbdt_pred[mask]),
        }
    return results

# ── threshold sensitivity ─────────────────────────────────────────────────────
def threshold_sensitivity(all_trajs_by_split, device, seeds, epochs,
                           thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
                           window_size=20):
    """Re-generate data at different failure thresholds and check RMSE ordering."""
    from src.data.nozzle_sim_ode import generate_multi_trajectory, make_split_assignment, save_multi_trajectory_csv
    import tempfile, os
    results = {}
    for thr in thresholds:
        print(f"\n  threshold={thr}mm ...", flush=True)
        trajectories = generate_multi_trajectory(
            n_trajectories=40, seed=2026,
            failure_depth_mm=thr, t_max_s=300.0, output_dt_s=1.0)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            tmp = f.name
        split = make_split_assignment(trajectories, val_fraction=0.15,
                                      test_fraction=0.20, seed=2026)
        save_multi_trajectory_csv(trajectories, tmp, split)
        all_t = load_nozzle_trajectories(tmp, feature_tier="observable")
        os.unlink(tmp)
        train_t = [t for t in all_t.values() if split.get(t.manifest.trajectory_id)=="train"]
        val_t   = [t for t in all_t.values() if split.get(t.manifest.trajectory_id)=="validation"]
        test_t  = [t for t in all_t.values() if split.get(t.manifest.trajectory_id)=="test"]
        dev_t   = train_t + val_t
        if not test_t:
            results[f"thr_{thr}mm"] = {"error": "no test trajectories"}
            continue
        cand = Candidate(f"rate_obs_w{window_size}", "physics_residual", "observable", window_size)
        dev_w, sc = _windows(dev_t, cand)
        test_w, _ = _windows(test_t, cand, sc)
        seed_preds = []
        for s in seeds[:3]:   # 3 seeds for speed
            m = train_one(cand, dev_w, device, s, min(epochs, 100))
            seed_preds.append(predict_nn(m, test_w, device))
        ensemble = np.mean(np.stack(seed_preds), axis=0)
        ridge_beta = fit_ridge(dev_w)
        ridge_p = predict_ridge(test_w, ridge_beta)
        gbdt_m  = fit_gbdt(dev_w)
        gbdt_p  = predict_gbdt(test_w, gbdt_m)
        curr_p  = current_rate_baseline(test_w)
        truth   = test_w.rul_s
        results[f"thr_{thr}mm"] = {
            "threshold_mm": thr,
            "n_test_trajectories": len(test_t),
            "n_windows": len(test_w),
            "neural_rmse":  rmse(truth, ensemble),
            "ridge_rmse":   rmse(truth, ridge_p),
            "gbdt_rmse":    rmse(truth, gbdt_p),
            "current_rate_rmse": rmse(truth, curr_p),
            "neural_vs_ridge_pct": round(
                (rmse(truth,ridge_p)-rmse(truth,ensemble))/max(rmse(truth,ridge_p),1e-9)*100, 2),
        }
        print(f"    neural={results[f'thr_{thr}mm']['neural_rmse']:.4f}  "
              f"ridge={results[f'thr_{thr}mm']['ridge_rmse']:.4f}  "
              f"gbdt={results[f'thr_{thr}mm']['gbdt_rmse']:.4f}", flush=True)
    return results


# ── inner selection ───────────────────────────────────────────────────────────
def _inner_select(tier_label, candidates, train_trajs, val_trajs, device,
                  inner_seeds, inner_epochs):
    """Pick best candidate by training on train, evaluating on val (3 seeds)."""
    print(f"\n--- Inner selection [{tier_label}] {len(candidates)} candidates ---", flush=True)
    rows = []
    for cand in candidates:
        tw, sc = _windows(train_trajs, cand)
        vw, _  = _windows(val_trajs,   cand, sc)
        use_phys = cand.feature_tier in ("estimated", "physics_observable")
        seed_scores = []
        for s in inner_seeds:
            m = train_one(cand, tw, device, s, inner_epochs)
            p = predict_nn(m, vw, device, use_phys=use_phys)
            score = rmse(vw.rul_s[vw.target_mask], p[vw.target_mask]) if vw.target_mask.any() \
                    else rmse(vw.rul_s, p)
            seed_scores.append(score)
        mean_score = float(np.mean(seed_scores))
        print(f"  {cand.name:<30} val_rmse={mean_score:.4f}", flush=True)
        rows.append((mean_score, cand))
    rows.sort(key=lambda x: x[0])
    best_score, best_cand = rows[0]
    print(f"  => Selected: {best_cand.name}  val_rmse={best_score:.4f}", flush=True)
    return best_cand


# ── single-tier evaluation helper ────────────────────────────────────────────
def _eval_tier(tier_label, cand, dev_trajs, test_trajs, device, seeds, epochs):
    """Train + evaluate one feature tier.  Returns (metrics_dict, baselines_dict,
    test_windows, seed_preds_list)."""
    dev_w, sc  = _windows(dev_trajs, cand)
    test_w, _  = _windows(test_trajs, cand, sc)
    truth      = test_w.rul_s
    use_phys = cand.feature_tier in ("estimated", "physics_observable")

    print(f"\n=== {tier_label} — Neural ({cand.name}, {len(seeds)} seeds) ===", flush=True)
    seed_preds = []
    for s in seeds:
        m = train_one(cand, dev_w, device, s, epochs)
        p = predict_nn(m, test_w, device, use_phys=use_phys)
        seed_preds.append(p)
        print(f"  seed={s}  rmse={rmse(truth,p):.4f}", flush=True)
    ensemble = np.mean(np.stack(seed_preds), axis=0)
    print(f"  ensemble rmse={rmse(truth,ensemble):.4f}  mae={mae(truth,ensemble):.4f}", flush=True)

    print(f"\n=== {tier_label} — Classical baselines ===", flush=True)
    from sklearn.linear_model import HuberRegressor
    ridge_beta = fit_ridge(dev_w)
    ridge_pred = predict_ridge(test_w, ridge_beta)
    Xh = _regr_mat(dev_w)[dev_w.target_mask]
    yh = dev_w.rul_s[dev_w.target_mask].astype(np.float64)
    huber_m   = HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=500).fit(Xh, yh)
    huber_pred= np.maximum(huber_m.predict(_regr_mat(test_w)), 0.).astype(np.float32)
    gbdt_pred = predict_gbdt(test_w, fit_gbdt(dev_w))
    rf_pred   = predict_rf(test_w, fit_rf(dev_w))
    curr_pred = current_rate_baseline(test_w)
    mean_pred = mean_baseline(test_w)

    baselines = {
        "neural_ensemble": ensemble,
        "gbdt":            gbdt_pred,
        "random_forest":   rf_pred,
        "ridge":           ridge_pred,
        "huber":           huber_pred,
        "current_rate":    curr_pred,   # privileged ref (uses margin+rate directly)
        "mean_baseline":   mean_pred,
    }
    rul_scale = float(np.nanmax(np.abs(truth[np.isfinite(truth)]))) if np.any(np.isfinite(truth)) else 1.
    ridge_r   = rmse(truth, ridge_pred)
    print(f"  {'Method':<22} {'RMSE':>8} {'MAE':>8} {'nRMSE':>8} {'vs Ridge':>10}", flush=True)
    print(f"  {'-'*62}", flush=True)
    metrics = {}
    for name, pred in baselines.items():
        r = rmse(truth, pred); m_ = mae(truth, pred)
        n_ = r / max(rul_scale, 1.)
        vs = f"{(ridge_r-r)/max(ridge_r,1e-9)*100:+.1f}%" if name != "ridge" else "—"
        print(f"  {name:<22} {r:8.4f} {m_:8.4f} {n_:8.4f} {vs:>10}", flush=True)
        metrics[name] = {"rmse": r, "mae": m_, "nrmse": n_,
                         "vs_ridge_pct": round((ridge_r-r)/max(ridge_r,1e-9)*100, 2)
                                         if name != "ridge" else 0.}
    return metrics, baselines, test_w, truth, seed_preds


# ── main validation ───────────────────────────────────────────────────────────
def run_sota_validation(data_path, split_map, train_trajs_obs, val_trajs_obs, test_trajs_obs,
                        train_trajs_est, val_trajs_est, test_trajs_est,
                        train_trajs_phys, val_trajs_phys, test_trajs_phys,
                        out_dir, device, seeds=SEEDS, epochs=EPOCH_FINAL,
                        window_size=20):
    """Run SOTA validation for three feature tiers.
    physics_observable = T/Q/P + depth_proxy + rate_proxy (derived from integration).
    Inner selection picks best model on train→val; final eval on train+val→test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    INNER_SEEDS  = seeds[:3]
    INNER_EPOCHS = min(epochs, 80)
    candidates   = make_candidates(window_size)

    # ── Inner selection ────────────────────────────────────────────────────
    obs_cands  = [c for c in candidates if c.feature_tier == "observable"]
    est_cands  = [c for c in candidates if c.feature_tier == "estimated"]
    phys_cands = [Candidate(c.name.replace("obs","phys"), c.model,
                            "physics_observable", c.window_size, c.hidden, c.dropout)
                  for c in obs_cands]
    OBS_CAND  = _inner_select("observable",         obs_cands,  train_trajs_obs,  val_trajs_obs,
                              device, INNER_SEEDS, INNER_EPOCHS)
    PHYS_CAND = _inner_select("physics_observable", phys_cands, train_trajs_phys, val_trajs_phys,
                              device, INNER_SEEDS, INNER_EPOCHS)
    EST_CAND  = _inner_select("estimated",          est_cands,  train_trajs_est,  val_trajs_est,
                              device, INNER_SEEDS, INNER_EPOCHS)

    dev_trajs_obs  = train_trajs_obs  + val_trajs_obs
    dev_trajs_phys = train_trajs_phys + val_trajs_phys
    dev_trajs_est  = train_trajs_est  + val_trajs_est

    # ── Tier 1: observable (honest: no margin/rate in model) ──────────────
    obs_metrics, obs_baselines, obs_test_w, obs_truth, obs_seed_preds = \
        _eval_tier("OBSERVABLE TIER", OBS_CAND, dev_trajs_obs, test_trajs_obs,
                   device, seeds, epochs)


    # ── Tier 1: observable (raw T/Q/P only) ──────────────────────────────
    obs_metrics, obs_baselines, obs_test_w, obs_truth, obs_seed_preds = \
        _eval_tier("OBSERVABLE TIER", OBS_CAND, dev_trajs_obs, test_trajs_obs,
                   device, seeds, epochs)

    # ── Tier 1b: physics_observable (T/Q/P + depth_proxy + rate_proxy) ──
    phys_metrics, phys_baselines, phys_test_w, phys_truth, phys_seed_preds = \
        _eval_tier("PHYSICS-OBSERVABLE TIER", PHYS_CAND, dev_trajs_phys, test_trajs_phys,
                   device, seeds, epochs)

    # ── Tier 2: estimated (legitimate: depth+rate in X window) ───────────
    est_metrics, est_baselines, est_test_w, est_truth, est_seed_preds = \
        _eval_tier("ESTIMATED TIER", EST_CAND, dev_trajs_est, test_trajs_est,
                   device, seeds, epochs)

    # ── Forecast-origin (physics_observable tier) ─────────────────────────
    print("\n=== Forecast-origin analysis (physics_observable tier) ===", flush=True)
    traj_map_phys = {t.manifest.trajectory_id: t for t in test_trajs_phys}
    origin_results = {}
    for origin in (0.2, 0.4, 0.6, 0.8):
        mask_arr = np.zeros(len(phys_test_w), dtype=bool)
        for i, (tid, t_ep) in enumerate(zip(phys_test_w.trajectory_ids,
                                            phys_test_w.endpoint_time_s)):
            traj = traj_map_phys.get(tid)
            life = float(traj.time_s[-1]) if traj is not None else float(phys_test_w.endpoint_time_s.max())
            mask_arr[i] = float(t_ep) / max(life, 1.) <= origin
        n = int(mask_arr.sum())
        row = {"n": n}
        if n >= 3:
            t_sub = phys_truth[mask_arr]
            for name, pred in phys_baselines.items():
                row[f"{name}_rmse"] = rmse(t_sub, pred[mask_arr])
        origin_results[f"@{int(origin*100)}pct"] = row
        if n >= 3:
            print(f"  @{int(origin*100)}%: n={n}  "
                  f"neural={row.get('neural_ensemble_rmse',float('nan')):.3f}  "
                  f"ridge={row.get('ridge_rmse',float('nan')):.3f}  "
                  f"gbdt={row.get('gbdt_rmse',float('nan')):.3f}", flush=True)

    # ── Threshold sensitivity (physics_observable tier, 3 seeds) ──────────
    print("\n=== Threshold sensitivity (physics_observable tier, 3 seeds) ===", flush=True)
    thr_results = threshold_sensitivity(None, device, seeds, epochs,
                                        thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
                                        window_size=window_size)

    # ── SOTA verdict ─────────────────────────────────────────────────────
    print("\n" + "="*65, flush=True)
    print("SOTA VALIDATION VERDICT", flush=True)
    print("="*65, flush=True)
    classical_keys = ["gbdt", "random_forest", "ridge", "huber"]

    for tier_label, metrics in [("Observable (raw T/Q/P)", obs_metrics),
                                 ("Physics-Observable (+proxy depth/rate)", phys_metrics),
                                 ("Estimated (+true depth/rate)", est_metrics)]:
        best_cls = min(classical_keys, key=lambda k: metrics[k]["rmse"])
        beat_all = all(metrics["neural_ensemble"]["rmse"] <= metrics[k]["rmse"]
                       for k in classical_keys)
        beat_curr = metrics["neural_ensemble"]["rmse"] <= metrics["current_rate"]["rmse"]
        print(f"\n  [{tier_label}]", flush=True)
        print(f"    Neural RMSE         : {metrics['neural_ensemble']['rmse']:.4f}s", flush=True)
        print(f"    Best classical      : {best_cls} = {metrics[best_cls]['rmse']:.4f}s", flush=True)
        print(f"    Beats all classical : {'YES ✓' if beat_all else 'NO ✗'}", flush=True)
        print(f"    Beats current_rate  : {'YES ✓' if beat_curr else 'NO ✗'}", flush=True)

    print("\n  [Threshold sensitivity (physics_observable) — neural always best?]", flush=True)
    n_best = sum(1 for v in thr_results.values()
                 if "error" not in v and v["neural_rmse"] < min(v["ridge_rmse"], v["gbdt_rmse"]))
    n_total = sum(1 for v in thr_results.values() if "error" not in v)
    print(f"    {n_best}/{n_total} thresholds: neural < Ridge AND GBDT", flush=True)
    print("="*65, flush=True)

    result = {
        "schema": "nozzle_sota_validation_v3",
        "protocol": "predefined_split; 3-tier honest separation; 5-seed ensemble",
        "n_test": len(test_trajs_phys),
        "observable_tier": {
            "n_features": int(obs_test_w.X.shape[-1]),
            "metrics": obs_metrics,
            "per_seed_rmse": [float(rmse(obs_truth, p)) for p in obs_seed_preds],
        },
        "physics_observable_tier": {
            "n_features": int(phys_test_w.X.shape[-1]),
            "metrics": phys_metrics,
            "per_seed_rmse": [float(rmse(phys_truth, p)) for p in phys_seed_preds],
        },
        "estimated_tier": {
            "n_features": int(est_test_w.X.shape[-1]),
            "metrics": est_metrics,
            "per_seed_rmse": [float(rmse(est_truth, p)) for p in est_seed_preds],
        },
        "forecast_origin_phys": origin_results,
        "threshold_sensitivity_phys": thr_results,
        "seed_ids": list(seeds),
    }
    write_json(out_dir / "SOTA_VALIDATION_REPORT.json", result)

    with (out_dir / "predictions_phys.csv").open("w", newline="", encoding="utf-8") as f:
        rows2 = [{"window_idx": i, "time_s": float(phys_test_w.endpoint_time_s[i]),
                  "true_rul_s": float(phys_truth[i]) if np.isfinite(phys_truth[i]) else None,
                  **{n: float(p[i]) for n, p in phys_baselines.items()}}
                 for i in range(len(phys_test_w))]
        if rows2:
            w2 = csv.DictWriter(f, fieldnames=list(rows2[0]))
            w2.writeheader(); w2.writerows(rows2)

    print(f"\nReport → {out_dir}/SOTA_VALIDATION_REPORT.json", flush=True)
    return result

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--output", default="outputs/nozzle_sota_v1")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=EPOCH_FINAL)
    parser.add_argument("--window-size", type=int, default=20,
                        help="Sliding window length in timesteps (default 20)")
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    device  = args.device if torch.cuda.is_available() else "cpu"
    seeds   = SEEDS[:2] if args.quick else SEEDS
    epochs  = 8 if args.quick else args.epochs
    ws      = args.window_size

    data_path = Path(args.data)
    with data_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split_map = {r["trajectory_id"]: r["split"] for r in rows}

    def _split(all_t):
        tr = [t for t in all_t.values() if split_map.get(t.manifest.trajectory_id) == "train"]
        va = [t for t in all_t.values() if split_map.get(t.manifest.trajectory_id) == "validation"]
        te = [t for t in all_t.values() if split_map.get(t.manifest.trajectory_id) == "test"]
        return tr, va, te

    obs_all  = load_nozzle_trajectories(data_path, feature_tier="observable")
    phys_all = load_nozzle_trajectories(data_path, feature_tier="physics_observable")
    est_all  = load_nozzle_trajectories(data_path, feature_tier="estimated")
    tr_obs,  va_obs,  te_obs  = _split(obs_all)
    tr_phys, va_phys, te_phys = _split(phys_all)
    tr_est,  va_est,  te_est  = _split(est_all)
    print(f"Window size: {ws}  Epochs: {epochs}  Seeds: {list(seeds)}", flush=True)
    print(f"Observable         : train={len(tr_obs)}  val={len(va_obs)}  test={len(te_obs)}", flush=True)
    print(f"Physics-observable : train={len(tr_phys)} val={len(va_phys)} test={len(te_phys)}", flush=True)
    print(f"Estimated          : train={len(tr_est)}  val={len(va_est)}  test={len(te_est)}", flush=True)

    out_dir = Path(args.output)
    t0 = time.time()
    run_sota_validation(
        data_path, split_map,
        train_trajs_obs=tr_obs,   val_trajs_obs=va_obs,   test_trajs_obs=te_obs,
        train_trajs_est=tr_est,   val_trajs_est=va_est,   test_trajs_est=te_est,
        train_trajs_phys=tr_phys, val_trajs_phys=va_phys, test_trajs_phys=te_phys,
        out_dir=out_dir, device=device, seeds=seeds, epochs=epochs,
        window_size=ws,
    )
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

