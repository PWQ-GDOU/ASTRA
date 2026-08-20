"""Reaction wheel v2 FULL: seq_len=30, all 5 candidates, 500-epoch budget.

Fixes the incomplete previous run (epochs=150, only 2 candidates).
Controlled comparison partner: exp_rw_v2_seq20.py (seq_len=20, otherwise identical).
"""
from __future__ import annotations

import argparse, csv, json, math, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.femto_strict import (
    BEARING_NAMES, FEATURE_GROUPS, FemtoSeries, FemtoWindowSet,
    fit_scaler, load_femto_zip, make_windows,
)
from src.models.reaction_wheel_strict import build_reaction_wheel_model, count_parameters
from scripts.rw_v2_common import (
    load_checkpoint,
    parse_csv_values,
    save_checkpoint,
    select_names,
    write_heartbeat,
)

SEEDS = (42, 123, 456, 2026, 3407)
COMMON_ENDPOINT = 19
SCHEDULER_EPOCH_BUDGET = 500
SEQ_LEN = 30

@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_group: str
    seq_len: int = SEQ_LEN

CANDIDATES = (
    Candidate("deep_ms_full_l30",  "deep_ms",   "full"),
    Candidate("deep_ms_trend_l30", "deep_ms",   "base_trend"),
    Candidate("large_gru_full_l30","large_gru",  "full"),
    Candidate("ms_full_l30",       "ms",         "full"),
    Candidate("gru_full_l30",      "gru",        "full"),
)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass

def rmse(yt, yp):
    yt = np.asarray(yt, dtype=np.float64).reshape(-1)
    yp = np.asarray(yp, dtype=np.float64).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sqrt(np.mean((yt[m]-yp[m])**2))) if np.any(m) else float("nan")

def _feature_names(group):
    return tuple(FEATURE_GROUPS[group])

def _append_cycle_pos(ws: FemtoWindowSet) -> FemtoWindowSet:
    age = ws.endpoints.astype(np.float32) / max(float(ws.rul_scale), 1.0)
    age_col = np.tile(age[:, None, None], (1, ws.X.shape[1], 1))
    return FemtoWindowSet(
        X=np.concatenate([ws.X, age_col], axis=-1).astype(np.float32),
        rul=ws.rul, life_fraction=ws.life_fraction, bearings=ws.bearings,
        endpoints=ws.endpoints,
        feature_names=ws.feature_names + ("cycle_pos_norm",),
        rul_scale=ws.rul_scale,
    )

def _window_sets(train_series, other_series, candidate, *, common_endpoint, rul_scale):
    names = _feature_names(candidate.feature_group)
    scaler = fit_scaler(train_series, names)
    train_set = make_windows(train_series, candidate.seq_len, scaler,
                             min_endpoint=common_endpoint, rul_scale=rul_scale)
    other_set = make_windows(other_series, candidate.seq_len, scaler,
                             min_endpoint=common_endpoint, rul_scale=rul_scale)
    return _append_cycle_pos(train_set), _append_cycle_pos(other_set), scaler

def train_model(candidate, train_set, val_set, device, seed, *,
                epochs, scheduler_epoch_budget, patience=50, batch_size=2048,
                lr=7e-4, heartbeat_path=None, heartbeat_context=None):
    seed_everything(seed)
    context = dict(heartbeat_context or {})
    write_heartbeat(
        heartbeat_path,
        phase="seed_start",
        candidate=context.get("candidate", candidate.name),
        seed=seed,
        epoch=0,
        epochs=epochs,
        **{key: value for key, value in context.items() if key != "candidate"},
    )
    model = build_reaction_wheel_model(candidate.model, train_set.X.shape[-1]).to(device)
    x = torch.as_tensor(train_set.X, dtype=torch.float32, device=device)
    target = torch.as_tensor(train_set.rul / train_set.rul_scale,
                             dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, max(epochs, scheduler_epoch_budget)
                   * math.ceil(max(len(train_set), 1) / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    val_x = val_y = None
    if val_set is not None:
        val_x = torch.as_tensor(val_set.X, dtype=torch.float32, device=device)
        val_y = val_set.rul
    best_state, best_score, best_epoch, stale = None, float("inf"), epochs, 0
    for epoch in range(1, epochs + 1):
        write_heartbeat(
            heartbeat_path,
            phase="epoch_start",
            candidate=context.get("candidate", candidate.name),
            seed=seed,
            epoch=epoch,
            epochs=epochs,
            **{key: value for key, value in context.items() if key != "candidate"},
        )
        model.train()
        order = torch.randperm(len(train_set), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            out = model(x.index_select(0, idx)).rul
            loss = F.smooth_l1_loss(
                out, target.index_select(0, idx), beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step()
        if val_x is not None:
            model.eval()
            with torch.no_grad():
                vp = model(val_x).rul.cpu().numpy() * train_set.rul_scale
            score = rmse(val_y, vp)
            if score < best_score:
                best_score = score; best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        write_heartbeat(
            heartbeat_path,
            phase="epoch_end",
            candidate=context.get("candidate", candidate.name),
            seed=seed,
            epoch=epoch,
            epochs=epochs,
            best_epoch=best_epoch,
            stale=stale,
            **{key: value for key, value in context.items() if key != "candidate"},
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_score


def predict(model, ws: FemtoWindowSet, device) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(ws.X, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x).rul.cpu().numpy() * ws.rul_scale


def _ridge_fit(train_set: FemtoWindowSet, alpha=1.0):
    X = train_set.X.reshape(len(train_set), -1)
    y = train_set.rul / train_set.rul_scale
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha).fit(X, y)


def _ridge_predict(reg, ws: FemtoWindowSet) -> np.ndarray:
    X = ws.X.reshape(len(ws), -1)
    return reg.predict(X) * ws.rul_scale


def run_loo(all_series, candidates, device, epochs, *, seeds, batch_size,
            patience, holdout_names, checkpoint_path, checkpoint_kwargs,
            initial_results, heartbeat_path=None):
    rul_scale = max(max(float(s.raw_rul[0]) for s in all_series), 1.0)
    results = {name: dict(values) for name, values in initial_results.items()}

    for holdout_name in holdout_names:
        train_series = [s for s in all_series if s.name != holdout_name]
        test_series  = [s for s in all_series if s.name == holdout_name]
        print(f"\n  holdout={holdout_name}", flush=True)
        write_heartbeat(
            heartbeat_path,
            phase="holdout_start",
            holdout=holdout_name,
            candidate=None,
        )

        fold_results = results.setdefault(holdout_name, {})
        for cand in candidates:
            if cand.name in fold_results:
                stored = fold_results[cand.name]
                print(
                    f"    [resume] {cand.name}: RMSE={stored['rmse']:.1f} "
                    f"nRMSE={stored['nrmse']:.4f}",
                    flush=True,
                )
                continue
            write_heartbeat(
                heartbeat_path,
                phase="candidate_start",
                holdout=holdout_name,
                candidate=cand.name,
                seeds=list(seeds),
            )
            train_set, test_set, _ = _window_sets(
                train_series, test_series, cand,
                common_endpoint=COMMON_ENDPOINT, rul_scale=rul_scale)
            val_series = [train_series[-1]]
            tr_series  = train_series[:-1]
            _, val_set, _ = _window_sets(
                tr_series, val_series, cand,
                common_endpoint=COMMON_ENDPOINT, rul_scale=rul_scale)
            seed_rmses = []
            for seed in seeds:
                model, best_ep, _ = train_model(
                    cand, train_set, val_set, device, seed,
                    epochs=epochs, scheduler_epoch_budget=epochs,
                    patience=patience, batch_size=batch_size,
                    heartbeat_path=heartbeat_path,
                    heartbeat_context={
                        "holdout": holdout_name,
                        "candidate": cand.name,
                    })
                preds = predict(model, test_set, device)
                seed_rmses.append(rmse(test_set.rul, preds))
            mean_r = float(np.mean(seed_rmses))
            nrmse  = mean_r / rul_scale
            fold_results[cand.name] = {
                "rmse": mean_r, "nrmse": nrmse, "seed_rmses": seed_rmses}
            save_checkpoint(
                checkpoint_path,
                rul_scale=rul_scale,
                per_fold=results,
                **checkpoint_kwargs,
            )
            print(f"    {cand.name}: RMSE={mean_r:.1f} nRMSE={nrmse:.4f}", flush=True)
            write_heartbeat(
                heartbeat_path,
                phase="candidate_end",
                holdout=holdout_name,
                candidate=cand.name,
                rmse=mean_r,
                nrmse=nrmse,
            )

        # Ridge baseline (full features, same seq_len)
        if "ridge_full" in fold_results:
            stored = fold_results["ridge_full"]
            print(
                f"    [resume] ridge_full: RMSE={stored['rmse']:.1f} "
                f"nRMSE={stored['nrmse']:.4f}",
                flush=True,
            )
            continue
        write_heartbeat(
            heartbeat_path,
            phase="ridge_start",
            holdout=holdout_name,
            candidate="ridge_full",
        )
        names_full = _feature_names("full")
        sc_full    = fit_scaler(train_series, names_full)
        tr_full    = make_windows(train_series, SEQ_LEN, sc_full,
                                  min_endpoint=COMMON_ENDPOINT, rul_scale=rul_scale)
        ts_full    = make_windows(test_series, SEQ_LEN, sc_full,
                                  min_endpoint=COMMON_ENDPOINT, rul_scale=rul_scale)
        tr_full = _append_cycle_pos(tr_full)
        ts_full = _append_cycle_pos(ts_full)
        reg         = _ridge_fit(tr_full)
        ridge_preds = _ridge_predict(reg, ts_full)
        ridge_r     = rmse(ts_full.rul, ridge_preds)
        fold_results["ridge_full"] = {
            "rmse": ridge_r, "nrmse": ridge_r / rul_scale, "seed_rmses": [ridge_r]}
        save_checkpoint(
            checkpoint_path,
            rul_scale=rul_scale,
            per_fold=results,
            **checkpoint_kwargs,
        )
        print(f"    ridge_full: RMSE={ridge_r:.1f} nRMSE={ridge_r/rul_scale:.4f}",
              flush=True)
        write_heartbeat(
            heartbeat_path,
            phase="ridge_end",
            holdout=holdout_name,
            candidate="ridge_full",
            rmse=ridge_r,
            nrmse=ridge_r / rul_scale,
        )

    return results, rul_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="outputs/reaction_wheel_v2_full")
    parser.add_argument("--device",   default="cuda:0")
    parser.add_argument("--epochs",   type=int, default=SCHEDULER_EPOCH_BUDGET)
    parser.add_argument("--data-zip", default="data/processed/femto_bearing.zip")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--holdouts", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--heartbeat", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing an implicit CPU fallback")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable CUDA device: {args.device}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_by_name = {candidate.name: candidate for candidate in CANDIDATES}
    candidate_names = select_names(
        candidate_by_name,
        parse_csv_values(args.candidates),
        label="candidates",
    )
    selected_candidates = tuple(candidate_by_name[name] for name in candidate_names)
    all_holdouts = tuple(BEARING_NAMES)
    holdout_names = select_names(
        all_holdouts,
        parse_csv_values(args.holdouts),
        label="holdouts",
    )
    seed_tokens = select_names(
        tuple(str(seed) for seed in SEEDS),
        parse_csv_values(args.seeds),
        label="seeds",
    )
    seeds = tuple(int(seed) for seed in seed_tokens)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        out_dir / "RW_V2_FULL_CHECKPOINT.json"
    )
    checkpoint_kwargs = {
        "experiment_schema": "reaction_wheel_v2_full",
        "seq_len": SEQ_LEN,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seeds": seeds,
        "candidate_names": candidate_names,
        "holdout_names": holdout_names,
    }
    initial_results = (
        load_checkpoint(checkpoint_path, **checkpoint_kwargs)
        if args.resume else {}
    )

    print(f"RW-v2-FULL  device={device}  epochs={args.epochs}  seq_len={SEQ_LEN}",
          flush=True)
    print(f"Candidates: {list(candidate_names)}", flush=True)
    print(f"Holdouts: {list(holdout_names)}  seeds={list(seeds)}", flush=True)
    print(f"Checkpoint: {checkpoint_path}  resume={args.resume}", flush=True)
    heartbeat_path = Path(args.heartbeat) if args.heartbeat else None
    write_heartbeat(
        heartbeat_path,
        phase="starting",
        device=str(device),
        seq_len=SEQ_LEN,
        epochs=args.epochs,
        candidates=list(candidate_names),
        holdouts=list(holdout_names),
    )

    t0 = time.time()
    all_series = list(load_femto_zip(str(Path(args.data_zip)), feature_group="full").values())
    print(f"Loaded {len(all_series)} bearings: {[s.name for s in all_series]}",
          flush=True)
    write_heartbeat(
        heartbeat_path,
        phase="data_loaded",
        device=str(device),
        bearings=len(all_series),
    )

    results, rul_scale = run_loo(
        all_series,
        selected_candidates,
        device,
        args.epochs,
        seeds=seeds,
        batch_size=args.batch_size,
        patience=args.patience,
        holdout_names=holdout_names,
        checkpoint_path=checkpoint_path,
        checkpoint_kwargs=checkpoint_kwargs,
        initial_results=initial_results,
        heartbeat_path=heartbeat_path,
    )

    print(f"\n{'='*60}", flush=True)
    print("RW-v2-FULL — LOO macro results", flush=True)
    print(f"{'='*60}", flush=True)
    cand_names = list(candidate_names) + ["ridge_full"]
    macro = {}
    for cn in cand_names:
        nrmses = [results[b][cn]["nrmse"] for b in holdout_names]
        macro[cn] = float(np.mean(nrmses))
        print(f"  {cn:30s}: macro nRMSE={macro[cn]:.4f}", flush=True)

    best_name = min(macro, key=macro.get)
    print(f"\nBest: {best_name}  macro nRMSE={macro[best_name]:.4f}", flush=True)
    elapsed = (time.time() - t0) / 60
    print(f"Total elapsed: {elapsed:.1f} min", flush=True)

    report = {
        "schema":         "reaction_wheel_v2_full",
        "seq_len":        SEQ_LEN,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "patience":       args.patience,
        "seeds":          list(seeds),
        "holdouts":       list(holdout_names),
        "rul_scale":      rul_scale,
        "candidates":     [asdict(c) for c in selected_candidates],
        "macro_nrmse":    macro,
        "best_candidate": best_name,
        "elapsed_min":    round(elapsed, 1),
        "per_fold":       results,
    }
    (out_dir / "RW_V2_FULL_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")

    with open(out_dir / "rw_v2_full_macro.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "macro_nrmse"])
        for cn, v in sorted(macro.items(), key=lambda x: x[1]):
            w.writerow([cn, f"{v:.4f}"])

    print(f"Report -> {out_dir}/RW_V2_FULL_REPORT.json", flush=True)
    write_heartbeat(
        heartbeat_path,
        phase="completed",
        report=str(out_dir / "RW_V2_FULL_REPORT.json"),
    )


if __name__ == "__main__":
    main()
