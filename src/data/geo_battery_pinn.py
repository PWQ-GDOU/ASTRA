"""Generate synthetic multi-trajectory GEO satellite battery degradation data.

Equations from simulation/geo_battery_pinn_extreme/ branch (COMSOL model).
Varies parameters via Latin Hypercube to produce independent trajectories.

Physics model (documented in governing_equations.md):
  d(soh)/dt   = -kdeg * sigma
  d(rint)/dt  = Rbase * 2.8 * kdeg * sigma
  d(Tcell)/dt = (Ttarget - Tcell) / tau_thermal

  sigma = clamp((0.20 + 0.92*F_DOD^1.35 + 0.45*aging)*alpha_T*f, 0, 5)
  F_DOD = 0.8 * shadow_min / 72
  shadow_min = 5 + 67*sin(pi*phase46)^0.65
  alpha_T = exp(0.048*(clamp(Tcell,-20,60)-25))
  f = 1 + 1.35*exp(-N/18)
  Ttarget = Tamb + 30 * (Idis^2 * rint * (0.36 + F_DOD))
  Tamb = 25 + 0.35*sin(2*pi*phase46)
  RUL = max((soh - 0.8) / (kdeg * sigma), 0)  [cycles]
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence


@dataclass
class GEOBatteryParams:
    """Physical and orbital parameters for one GEO satellite battery trajectory."""
    trajectory_id: str
    kdeg: float = 2.15e-4          # degradation coefficient
    Rbase: float = 0.034           # initial internal resistance [Ohm]
    Qnom_mAh: float = 2000.0       # nominal capacity [mAh]
    Idis: float = 1.334            # discharge current [A] = 0.667C * 2Ah
    failSOH: float = 0.80          # failure SOH threshold
    orbit_phase_offset: float = 0.0  # orbit phase offset [0,1) for variability
    tau_thermal: float = 18.0       # thermal time constant [s]
    initial_soh: float = 1.0       # initial state of health
    initial_Tcell: float = 25.0    # initial cell temperature [°C]


def simulate_geo_battery(
    params: GEOBatteryParams,
    n_cycles: int = 1200,
) -> dict[str, np.ndarray]:
    """Integrate GEO battery degradation ODE for n_cycles."""
    soh = float(params.initial_soh)
    rint = float(params.Rbase)
    Tcell = float(params.initial_Tcell)

    records = {
        "cycle": np.zeros(n_cycles, dtype=np.float64),
        "soh": np.zeros(n_cycles),
        "capacity_mAh": np.zeros(n_cycles),
        "rint": np.zeros(n_cycles),
        "Tcell": np.zeros(n_cycles),
        "shadow_min": np.zeros(n_cycles),
        "F_DOD": np.zeros(n_cycles),
        "stress": np.zeros(n_cycles),
        "RUL": np.zeros(n_cycles),
        "formation": np.zeros(n_cycles),
        "alpha_T": np.zeros(n_cycles),
        "Tamb": np.zeros(n_cycles),
    }

    for n in range(n_cycles):
        phase46 = ((n / 46.0) + params.orbit_phase_offset) % 1.0
        shadow_min = 5.0 + 67.0 * max(0.0, np.sin(np.pi * phase46)) ** 0.65
        F_DOD = 0.8 * shadow_min / 72.0
        Tamb = 25.0 + 0.35 * np.sin(2.0 * np.pi * phase46)
        aging = 1.0 - soh
        Tcell_clamped = float(np.clip(Tcell, -20.0, 60.0))
        alpha_T = float(np.exp(0.048 * (Tcell_clamped - 25.0)))
        formation = 1.0 + 1.35 * np.exp(-n / 18.0)
        stress_raw = (0.20 + 0.92 * F_DOD ** 1.35 + 0.45 * aging) * alpha_T * formation
        stress = float(np.clip(stress_raw, 0.0, 5.0))
        Qheat = params.Idis ** 2 * rint * (0.36 + F_DOD)
        Ttarget = Tamb + 30.0 * Qheat
        # Forward Euler integration (1 cycle per step)
        soh = max(soh - params.kdeg * stress, 0.0)
        rint = float(rint + params.Rbase * 2.8 * params.kdeg * stress)
        Tcell = float(Tcell + (Ttarget - Tcell) / params.tau_thermal)
        rul = max((soh - params.failSOH) / max(params.kdeg * stress, 1e-12), 0.0)
        records["cycle"][n] = n
        records["soh"][n] = soh
        records["capacity_mAh"][n] = params.Qnom_mAh * soh
        records["rint"][n] = rint
        records["Tcell"][n] = Tcell
        records["shadow_min"][n] = shadow_min
        records["F_DOD"][n] = F_DOD
        records["stress"][n] = stress
        records["RUL"][n] = rul
        records["formation"][n] = formation
        records["alpha_T"][n] = alpha_T
        records["Tamb"][n] = Tamb
        if soh <= params.failSOH:
            # Truncate at first failure
            for key in records:
                records[key] = records[key][:n + 1]
            break
    return records


def lhc_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling: n×1 uniform samples."""
    perm = rng.permutation(n)
    return (perm + rng.uniform(size=n)) / n


def generate_multi_trajectory(
    n_trajectories: int = 30,
    seed: int = 42,
    *,
    kdeg_range: tuple[float, float] = (1.5e-4, 3.0e-4),
    orbit_phase_range: tuple[float, float] = (0.0, 1.0),
    initial_soh_range: tuple[float, float] = (0.98, 1.00),
    tau_thermal_range: tuple[float, float] = (12.0, 24.0),
) -> list[tuple[GEOBatteryParams, dict[str, np.ndarray]]]:
    """Generate trajectories via Latin Hypercube Sampling over key parameters."""
    rng = np.random.default_rng(seed)
    kdeg_samples = lhc_sample(n_trajectories, rng) * (kdeg_range[1] - kdeg_range[0]) + kdeg_range[0]
    phase_samples = lhc_sample(n_trajectories, rng) * (orbit_phase_range[1] - orbit_phase_range[0])
    soh_samples = lhc_sample(n_trajectories, rng) * (initial_soh_range[1] - initial_soh_range[0]) + initial_soh_range[0]
    tau_samples = lhc_sample(n_trajectories, rng) * (tau_thermal_range[1] - tau_thermal_range[0]) + tau_thermal_range[0]
    results = []
    for i in range(n_trajectories):
        params = GEOBatteryParams(
            trajectory_id=f"GEO_T{i + 1:03d}",
            kdeg=float(kdeg_samples[i]),
            orbit_phase_offset=float(phase_samples[i]),
            initial_soh=float(soh_samples[i]),
            tau_thermal=float(tau_samples[i]),
        )
        records = simulate_geo_battery(params)
        results.append((params, records))
    return results


def save_multi_trajectory_csv(
    trajectories: list[tuple[GEOBatteryParams, dict[str, np.ndarray]]],
    output_path: str | Path,
    split_assignment: dict[str, str] | None = None,
) -> None:
    """Save all trajectories to a single multi-trajectory CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trajectory_id", "split", "cycle",
        "soh", "capacity_mAh", "rint", "Tcell",
        "shadow_min", "F_DOD", "stress", "RUL",
        "formation", "alpha_T", "Tamb",
        "kdeg", "orbit_phase_offset", "initial_soh", "tau_thermal",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for params, records in trajectories:
            split = (split_assignment or {}).get(params.trajectory_id, "train")
            n = len(records["cycle"])
            for i in range(n):
                row = {
                    "trajectory_id": params.trajectory_id,
                    "split": split,
                    "cycle": int(records["cycle"][i]),
                    "soh": float(records["soh"][i]),
                    "capacity_mAh": float(records["capacity_mAh"][i]),
                    "rint": float(records["rint"][i]),
                    "Tcell": float(records["Tcell"][i]),
                    "shadow_min": float(records["shadow_min"][i]),
                    "F_DOD": float(records["F_DOD"][i]),
                    "stress": float(records["stress"][i]),
                    "RUL": float(records["RUL"][i]),
                    "formation": float(records["formation"][i]),
                    "alpha_T": float(records["alpha_T"][i]),
                    "Tamb": float(records["Tamb"][i]),
                    "kdeg": params.kdeg,
                    "orbit_phase_offset": params.orbit_phase_offset,
                    "initial_soh": params.initial_soh,
                    "tau_thermal": params.tau_thermal,
                }
                writer.writerow(row)


def make_split_assignment(
    trajectories: list[tuple[GEOBatteryParams, dict]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 2026,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    ids = [params.trajectory_id for params, _ in trajectories]
    perm = rng.permutation(len(ids))
    n_test = max(1, int(len(ids) * test_fraction))
    n_val = max(1, int(len(ids) * val_fraction))
    assignment = {}
    for i, idx in enumerate(perm):
        if i < n_test:
            assignment[ids[idx]] = "test"
        elif i < n_test + n_val:
            assignment[ids[idx]] = "validation"
        else:
            assignment[ids[idx]] = "train"
    return assignment


def summarise(trajectories, split_assignment):
    counts = {"train": 0, "validation": 0, "test": 0}
    for params, records in trajectories:
        split = split_assignment.get(params.trajectory_id, "train")
        counts[split] += 1
    life_lengths = [len(records["cycle"]) for _, records in trajectories]
    rul_at_start = [float(records["RUL"][0]) for _, records in trajectories]
    return {
        "n_trajectories": len(trajectories),
        "split_counts": counts,
        "life_length_mean": float(np.mean(life_lengths)),
        "life_length_range": [int(min(life_lengths)), int(max(life_lengths))],
        "rul_at_start_mean": float(np.mean(rul_at_start)),
        "rul_at_start_range": [float(min(rul_at_start)), float(max(rul_at_start))],
        "all_event_observed": all(float(records["soh"][-1]) <= 0.80 for _, records in trajectories),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/geo_battery_pinn")
    args = parser.parse_args()
    out = Path(args.output)
    trajectories = generate_multi_trajectory(n_trajectories=args.n, seed=args.seed)
    split_assignment = make_split_assignment(trajectories)
    save_multi_trajectory_csv(trajectories, out / "geo_battery_multi_trajectory.csv", split_assignment)
    summary = summarise(trajectories, split_assignment)
    meta = {
        "schema": "geo_battery_pinn_multi_trajectory_v1",
        "source": "Generated from COMSOL GEO battery PINN equations (simulation/geo_battery_pinn_extreme branch)",
        "n_trajectories": args.n,
        "seed": args.seed,
        "lhc_parameters": ["kdeg", "orbit_phase_offset", "initial_soh", "tau_thermal"],
        "split_assignment": split_assignment,
        **summary,
    }
    (out / "geo_battery_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out}")
