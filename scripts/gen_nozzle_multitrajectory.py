"""Generate multi-condition nozzle ablation trajectories for the benchmark.

Runs the reduced-order ODE simulator (src/data/nozzle_sim_ode.py) with
Latin-Hypercube-sampled physical parameters and saves the result CSV and
metadata JSON.  Designed to be executed on the remote GPU server.

Usage
-----
  python scripts/gen_nozzle_multitrajectory.py \\
      --n-traj 40 \\
      --output data/processed/nozzle_multitrajectory \\
      --seed 2026

Smoke test (no GPU needed):
  python scripts/gen_nozzle_multitrajectory.py --n-traj 6 --output /tmp/nozzle_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nozzle_sim_ode import (
    generate_multi_trajectory,
    make_split_assignment,
    save_metadata_json,
    save_multi_trajectory_csv,
    simulate_nozzle,
    NozzleConditionParams,
    _K_AB,
    _TAU_TH,
    _N_TH,
    _H_REF,
)


def print_summary(trajectories, split_assignment):
    print("\n=== Trajectory summary ===")
    event_cnt = sum(1 for _, r in trajectories if r["reached_failure"])
    censored = len(trajectories) - event_cnt
    lives = [r["time_s"][-1] for _, r in trajectories]
    print(f"  Total:        {len(trajectories)}")
    print(f"  Run-to-failure: {event_cnt}  ({100*event_cnt/len(trajectories):.0f}%)")
    print(f"  Right-censored: {censored}")
    print(f"  Life range:   {min(lives):.1f} – {max(lives):.1f} s")
    print(f"  Life median:  {float(np.median(lives)):.1f} s")

    cond_counts = Counter(p.condition_id for p, _ in trajectories)
    print(f"  Conditions:   {len(cond_counts)} distinct")
    for cid, cnt in sorted(cond_counts.items()):
        print(f"    {cid}: {cnt} trajectories")

    split_counts = Counter(split_assignment.values())
    print(f"  Split:  {dict(split_counts)}")

    # Calibration check: reference run at 20s
    ref = NozzleConditionParams("_ref", "_ref")
    from src.data.nozzle_sim_ode import simulate_nozzle
    r = simulate_nozzle(ref, output_dt_s=1.0)
    idx = min(20, len(r["time_s"]) - 1)
    print(f"\n  Calibration check (reference run):")
    print(f"    depth(20s) = {r['cumulative_ablation_depth_mm'][idx]:.4f} mm  (COMSOL target 0.2585)")
    print(f"    T_solid(20s) = {r['solid_temperature_K'][idx]:.1f} K  (COMSOL target 946.2)")
    print(f"    k_ab = {_K_AB:.4e}  τ_th = {_TAU_TH}  n_th = {_N_TH}  h_ref = {_H_REF}")


def main():
    parser = argparse.ArgumentParser(description="Generate nozzle multi-trajectory dataset")
    parser.add_argument("--n-traj", type=int, default=40,
                        help="Number of trajectories to generate (default 40)")
    parser.add_argument("--output", default="data/processed/nozzle_multitrajectory",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed for LHC sampling")
    parser.add_argument("--failure-depth-mm", type=float, default=0.50,
                        help="Ablation depth failure threshold in mm (default 0.5)")
    parser.add_argument("--t-max-s", type=float, default=300.0,
                        help="Maximum simulation time in seconds (default 300)")
    parser.add_argument("--output-dt-s", type=float, default=1.0,
                        help="Time step between recorded output rows (default 1.0 s)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_traj} trajectories  seed={args.seed}  "
          f"failure_depth={args.failure_depth_mm}mm  t_max={args.t_max_s}s")
    t0 = time.time()

    trajectories = generate_multi_trajectory(
        n_trajectories=args.n_traj,
        seed=args.seed,
        failure_depth_mm=args.failure_depth_mm,
        t_max_s=args.t_max_s,
        output_dt_s=args.output_dt_s,
    )

    split_assignment = make_split_assignment(
        trajectories,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    csv_path = out_dir / f"nozzle_sim_{args.n_traj}traj.csv"
    meta_path = out_dir / f"nozzle_sim_{args.n_traj}traj_metadata.json"

    save_multi_trajectory_csv(trajectories, csv_path, split_assignment)
    save_metadata_json(trajectories, meta_path, split_assignment)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  →  {csv_path}")
    print_summary(trajectories, split_assignment)

    # Write run provenance
    provenance = {
        "schema": "nozzle_gen_provenance_v1",
        "n_trajectories": args.n_traj,
        "seed": args.seed,
        "failure_depth_mm": args.failure_depth_mm,
        "t_max_s": args.t_max_s,
        "output_dt_s": args.output_dt_s,
        "elapsed_s": elapsed,
        "csv": str(csv_path),
        "event_count": sum(1 for _, r in trajectories if r["reached_failure"]),
        "censored_count": sum(1 for _, r in trajectories if not r["reached_failure"]),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
