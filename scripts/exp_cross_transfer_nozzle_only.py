"""Run ONLY the nozzle-source cross-transfer LOO (fixed column mapping) and
merge with the already-completed GEO_battery results, instead of re-running
the ~11h GEO battery LOO loop again.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/data/disk3/spacecraft_rul")
sys.path.insert(0, "/data/disk3/spacecraft_rul/scripts")

from exp_cross_transfer_v2fix import (
    load_femto_early, load_nozzle_source, run_cross_transfer_loo,
    SEQ_LEN, EARLY_FRAC, COMMON_EP, N_SHOTS, N_TRIALS, SEEDS,
)
from src.data.femto_strict import load_femto_zip
import numpy as np

device = "cpu"
out_dir = Path("/data/disk3/spacecraft_rul/outputs/cross_component_transfer_v2fix")
prev_report_path = out_dir / "CROSS_TRANSFER_REPORT.json"
prev_report = json.loads(prev_report_path.read_text(encoding="utf-8"))
assert "GEO_battery" in prev_report["results"], "expected existing GEO_battery results to merge"

t0 = time.time()
print("Loading FEMTO bearing data...", flush=True)
series_dict = load_femto_zip("/data/disk3/spacecraft_rul/data/processed/femto_bearing.zip", feature_group="base")
all_series = list(series_dict.values())
rul_scale = max(max(float(s.raw_rul[0]) for s in all_series), 1.0)
bearing_data, rul_scale = load_femto_early(all_series, SEQ_LEN, "base", COMMON_EP, EARLY_FRAC, rul_scale)
print(f"Loaded {len(bearing_data)} bearings  rul_scale={rul_scale:.0f}", flush=True)

print("\nLoading nozzle source (fixed multi-trajectory loader)...", flush=True)
Xn, yn, cpn = load_nozzle_source(SEQ_LEN, device=device)
assert Xn is not None, "nozzle source failed to load even after fix"
res_nozzle = run_cross_transfer_loo(Xn, yn, cpn, bearing_data, rul_scale, "nozzle", device)

all_results = dict(prev_report["results"])
all_results["nozzle"] = res_nozzle

print(f"\n{'='*70}", flush=True)
print("CROSS-COMPONENT TRANSFER — macro LOO results (merged)", flush=True)
print(f"{'='*70}", flush=True)
for src_name, src_res in all_results.items():
    print(f"\nSource: {src_name}", flush=True)
    for N in N_SHOTS:
        tr_list = [src_res[b][str(N)]["transfer_rmse"] if str(N) in src_res[b] else src_res[b].get(N, {}).get("transfer_rmse")
                   for b in src_res if (str(N) in src_res[b] or N in src_res[b])]
        sc_list = [src_res[b][str(N)]["scratch_rmse"] if str(N) in src_res[b] else src_res[b].get(N, {}).get("scratch_rmse")
                   for b in src_res if (str(N) in src_res[b] or N in src_res[b])]
        if not tr_list: continue
        t_mac = float(np.mean(tr_list)); s_mac = float(np.mean(sc_list))
        gain = (s_mac - t_mac) / max(s_mac, 1e-9) * 100
        print(f"  N={N:2d}: Transfer={t_mac:.1f}  Scratch={s_mac:.1f}  Gain={gain:+.1f}%", flush=True)

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)

report = dict(prev_report)
report["sources"] = list(all_results.keys())
report["results"] = all_results
report["nozzle_source_fix_note"] = (
    "nozzle source re-run after fixing load_nozzle_source() to use "
    "src.data.nozzle_multitrajectory (200-trajectory strict loader) instead "
    "of English-substring column matching against the Chinese-column single-"
    "trajectory CSV, which always failed and silently skipped nozzle. "
    "GEO_battery results below are unchanged, carried over from the prior run."
)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "CROSS_TRANSFER_REPORT.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Report -> {out_dir}/CROSS_TRANSFER_REPORT.json", flush=True)
