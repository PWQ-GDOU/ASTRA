"""Reduced-order nozzle ablation ODE simulator.

Calibrated on the COMSOL 20-second C/C throat ablation trajectory
(H₂O₂ / polyethylene hybrid, 1.43 MPa chamber pressure, 1147 K inlet gas).

Physics summary
---------------
Three coupled equations are integrated forward at each timestep:

1. Surface-temperature rise (semi-infinite wall with forced convection):
        T_s(t) = T_in - (T_in - T_s0) / (1 + t/τ_th)^n_th
   Exponent form approximates the sqrt-time heating of a thick thermal mass
   while bounding T_s to T_in.  τ_th and n_th are calibrated on COMSOL.

2. Convective heat flux (Dittus–Boelter scaling):
        Q(t) = h_eff × (T_in - T_s(t))
        h_eff = h_ref × mdot_f^0.8 × (dt0_ref/dt0_mm)^0.2

3. Oxidative ablation rate (Arrhenius-type; reduced to power-law in P and Q):
        r_dot(t) [m/s] = k_ab × k_ab_f × Q(t) × (P(t)/P_ref)^n_pres

   k_ab is calibrated so that the reference run gives exactly
   cumulative_depth(20 s) = 0.2585 mm (COMSOL target).

Condition space (LHC-sampled, 5 parameters)
--------------------------------------------
  T_in_K      : inlet gas temperature   [920, 1 320] K
  p_ch0_Pa    : initial chamber pressure [0.8, 2.0] MPa
  mdot_factor : mass-flow rate scale     [0.65, 1.35]
  k_ab_factor : ablation chemistry scale [0.70, 1.40]
  dt0_mm      : initial throat diameter  [12.0, 20.0] mm

Output CSV schema (matches nozzle_multitrajectory.py loader)
------------------------------------------------------------
trajectory_id, condition_id, time_s,
cumulative_ablation_depth_mm, ablation_rate_m_s, failure_depth_mm,
solid_temperature_K, heat_flux_W_m2, pressure_Pa
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# COMSOL calibration constants
# ---------------------------------------------------------------------------
_CALIB_T_IN: float = 1147.0       # K
_CALIB_P_CH: float = 1.43e6       # Pa
_CALIB_DEPTH_20S: float = 0.2585e-3  # m
_CALIB_T_S0: float = 300.0        # K
_CALIB_T_S_20S: float = 946.2     # K  (COMSOL final solid temp)

# Thermal relaxation (calibrated: T_s(20s) ≈ 946K under reference conditions)
_TAU_TH: float = 0.6    # s
_N_TH: float = 0.42

# Base heat transfer coefficient (calibrated from COMSOL Q/(T_in-T_s) values)
_H_REF: float = 2_500.0  # W/(m²·K)

# Pressure exponent for oxidative ablation chemistry
_N_PRES: float = 0.40

# Reference throat diameter
_DT0_REF_MM: float = 15.875  # mm


def _calibrate_k_ab(dt_s: float = 0.05) -> float:
    """Return k_ab [m³/J] so that reference run gives 0.2585 mm in 20 s."""
    T_in = _CALIB_T_IN
    T_s0 = _CALIB_T_S0
    total_Qdt = 0.0
    t = 0.0
    while t < 20.0 - 1e-9:
        t_mid = t + dt_s / 2.0
        T_s = T_in - (T_in - T_s0) / (1.0 + t_mid / _TAU_TH) ** _N_TH
        Q = _H_REF * max(T_in - T_s, 0.0)
        # at reference: p == P_ref → (P/P_ref)^n = 1
        total_Qdt += Q * dt_s
        t += dt_s
    return _CALIB_DEPTH_20S / total_Qdt   # m / (W·s/m²) = m³/J


_K_AB: float = _calibrate_k_ab()


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NozzleConditionParams:
    """Physical parameters for one simulation run."""
    trajectory_id: str
    condition_id: str
    T_in_K: float = _CALIB_T_IN          # inlet gas temperature, K
    p_ch0_Pa: float = _CALIB_P_CH        # initial chamber pressure, Pa
    mdot_factor: float = 1.0             # mass-flow rate scale
    k_ab_factor: float = 1.0             # ablation chemistry scale
    dt0_mm: float = _DT0_REF_MM          # initial throat diameter, mm
    T_s0_K: float = _CALIB_T_S0         # initial solid temperature, K
    failure_depth_mm: float = 0.50       # failure threshold, mm
    t_max_s: float = 300.0              # max simulation time, s
    p_decay_frac: float = 0.15          # fractional pressure drop by t=100s


# ---------------------------------------------------------------------------
# ODE integrator
# ---------------------------------------------------------------------------

def simulate_nozzle(
    params: NozzleConditionParams,
    dt_s: float = 1.0,
    output_dt_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Integrate the ablation ODE and return time-series arrays.

    Returns a dict with keys:
        time_s, cumulative_ablation_depth_mm, ablation_rate_m_s,
        solid_temperature_K, heat_flux_W_m2, pressure_Pa, failure_depth_mm
        reached_failure (bool scalar)
    """
    T_in = params.T_in_K
    T_s0 = params.T_s0_K

    # Condition-dependent heat-transfer coefficient (Dittus-Boelter + geometry)
    h_eff = (
        _H_REF
        * params.mdot_factor ** 0.8
        * (_DT0_REF_MM / params.dt0_mm) ** 0.2
    )

    # Thermal time-constant scales inversely with convective intensity
    tau_th = _TAU_TH / max(params.mdot_factor ** 0.5, 0.2)

    # Build output time grid (coarse output, fine inner step if needed)
    inner_dt = min(dt_s, 0.1)     # always integrate at ≤0.1 s
    out_interval = max(1, round(output_dt_s / inner_dt))

    # Don't include t=0 (rate=0 before first integration step; loader rejects ≤0)
    times: list[float] = []
    depths_mm: list[float] = []
    rates_ms: list[float] = []
    T_solids: list[float] = []
    heat_fluxes: list[float] = []
    pressures: list[float] = []

    depth_mm = 0.0
    t = 0.0
    step = 0
    reached_failure = False

    while t < params.t_max_s:
        t_new = t + inner_dt
        t_mid = 0.5 * (t + t_new)

        # --- solid temperature (semi-infinite thermal model) ---
        T_s = T_in - (T_in - T_s0) / (1.0 + t_mid / tau_th) ** _N_TH

        # --- convective heat flux ---
        Q = h_eff * max(T_in - T_s, 0.0)

        # --- chamber pressure (linear decay) ---
        p = params.p_ch0_Pa * (
            1.0 - params.p_decay_frac * min(t_mid, 100.0) / 100.0
        )

        # --- ablation rate ---
        r_dot = _K_AB * params.k_ab_factor * Q * (p / _CALIB_P_CH) ** _N_PRES

        # --- integrate depth ---
        depth_mm += r_dot * inner_dt * 1000.0  # m → mm

        t = t_new
        step += 1

        # record at output resolution
        if step % out_interval == 0:
            times.append(t)
            depths_mm.append(depth_mm)
            rates_ms.append(r_dot)
            T_solids.append(T_s)
            heat_fluxes.append(Q)
            pressures.append(p)

        if depth_mm >= params.failure_depth_mm:
            # Ensure last point is recorded
            if times[-1] < t - 1e-9:
                times.append(t)
                depths_mm.append(depth_mm)
                rates_ms.append(r_dot)
                T_solids.append(T_s)
                heat_fluxes.append(Q)
                pressures.append(p)
            reached_failure = True
            break

    n = len(times)
    return {
        "time_s": np.array(times, dtype=np.float64),
        "cumulative_ablation_depth_mm": np.array(depths_mm, dtype=np.float64),
        "ablation_rate_m_s": np.array(rates_ms, dtype=np.float64),
        "solid_temperature_K": np.array(T_solids, dtype=np.float64),
        "heat_flux_W_m2": np.array(heat_fluxes, dtype=np.float64),
        "pressure_Pa": np.array(pressures, dtype=np.float64),
        "failure_depth_mm": float(params.failure_depth_mm),
        "reached_failure": reached_failure,
        "n_steps": n,
    }


# ---------------------------------------------------------------------------
# Latin Hypercube sampling & multi-trajectory generation
# ---------------------------------------------------------------------------

def _lhc_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n,) array uniformly sampled in [0,1] via LHC."""
    cuts = np.arange(n, dtype=float) / n
    return rng.permutation(cuts + rng.random(n) / n)


# Condition parameter ranges (name, low, high)
_CONDITION_RANGES = {
    "T_in_K":      (920.0,  1320.0),
    "p_ch0_MPa":   (0.80,   2.00),
    "mdot_factor": (0.65,   1.35),
    "k_ab_factor": (0.70,   1.40),
    "dt0_mm":      (12.0,   20.0),
}

_N_COND_PARAMS = len(_CONDITION_RANGES)


def _make_condition_id(T_in: float, p_MPa: float, mdot: float) -> str:
    """Group trajectories into discrete conditions by binning key parameters."""
    T_bin = int((T_in - 920.0) / 100.0)     # 0-3
    p_bin = int((p_MPa - 0.8) / 0.40)       # 0-2
    m_bin = int((mdot - 0.65) / 0.35)       # 0-1
    return f"C{T_bin}{p_bin}{m_bin}"


def generate_multi_trajectory(
    n_trajectories: int = 40,
    seed: int = 2026,
    failure_depth_mm: float = 0.50,
    t_max_s: float = 300.0,
    output_dt_s: float = 1.0,
) -> list[tuple[NozzleConditionParams, dict]]:
    """Generate ``n_trajectories`` ablation trajectories via LHC sampling.

    Each trajectory uses a unique LHC parameter combination.  Trajectories
    that reach the failure threshold are marked *event_observed*; those that
    exceed ``t_max_s`` without failure are right-censored.

    Returns a list of (params, records) tuples where *records* is the dict
    returned by :func:`simulate_nozzle`.
    """
    rng = np.random.default_rng(seed)
    keys = list(_CONDITION_RANGES.keys())

    samples = np.column_stack([_lhc_sample(n_trajectories, rng) for _ in keys])  # (N, 5)

    results: list[tuple[NozzleConditionParams, dict]] = []
    for i in range(n_trajectories):
        vals = {}
        for j, key in enumerate(keys):
            lo, hi = _CONDITION_RANGES[key]
            vals[key] = lo + samples[i, j] * (hi - lo)

        cid = _make_condition_id(vals["T_in_K"], vals["p_ch0_MPa"], vals["mdot_factor"])
        params = NozzleConditionParams(
            trajectory_id=f"SIM_T{i + 1:03d}",
            condition_id=cid,
            T_in_K=vals["T_in_K"],
            p_ch0_Pa=vals["p_ch0_MPa"] * 1e6,
            mdot_factor=vals["mdot_factor"],
            k_ab_factor=vals["k_ab_factor"],
            dt0_mm=vals["dt0_mm"],
            failure_depth_mm=failure_depth_mm,
            t_max_s=t_max_s,
        )
        records = simulate_nozzle(params, output_dt_s=output_dt_s)
        results.append((params, records))

    return results


def make_split_assignment(
    trajectories: list[tuple[NozzleConditionParams, dict]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 2026,
) -> dict[str, str]:
    """Assign trajectory_ids to train/validation/test splits."""
    rng = np.random.default_rng(seed)
    ids = [p.trajectory_id for p, _ in trajectories]
    perm = rng.permutation(len(ids)).tolist()
    n_test = max(1, int(len(ids) * test_fraction))
    n_val = max(1, int(len(ids) * val_fraction))
    assignment: dict[str, str] = {}
    for rank, idx in enumerate(perm):
        if rank < n_test:
            assignment[ids[idx]] = "test"
        elif rank < n_test + n_val:
            assignment[ids[idx]] = "validation"
        else:
            assignment[ids[idx]] = "train"
    return assignment


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "trajectory_id", "condition_id", "split", "time_s",
    "cumulative_ablation_depth_mm", "ablation_rate_m_s", "failure_depth_mm",
    "solid_temperature_K", "heat_flux_W_m2", "pressure_Pa",
    "reached_failure",
    # Condition parameters (for audit)
    "T_in_K", "p_ch0_Pa", "mdot_factor", "k_ab_factor", "dt0_mm",
]


def save_multi_trajectory_csv(
    trajectories: list[tuple[NozzleConditionParams, dict]],
    output_path: str | Path,
    split_assignment: dict[str, str] | None = None,
) -> None:
    """Save all trajectories to a single CSV in the nozzle_multitrajectory schema."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for params, records in trajectories:
            split = (split_assignment or {}).get(params.trajectory_id, "train")
            n = len(records["time_s"])
            for i in range(n):
                writer.writerow({
                    "trajectory_id": params.trajectory_id,
                    "condition_id": params.condition_id,
                    "split": split,
                    "time_s": float(records["time_s"][i]),
                    "cumulative_ablation_depth_mm": float(records["cumulative_ablation_depth_mm"][i]),
                    "ablation_rate_m_s": float(records["ablation_rate_m_s"][i]),
                    "failure_depth_mm": float(records["failure_depth_mm"]),
                    "solid_temperature_K": float(records["solid_temperature_K"][i]),
                    "heat_flux_W_m2": float(records["heat_flux_W_m2"][i]),
                    "pressure_Pa": float(records["pressure_Pa"][i]),
                    "reached_failure": int(records["reached_failure"]),
                    "T_in_K": params.T_in_K,
                    "p_ch0_Pa": params.p_ch0_Pa,
                    "mdot_factor": params.mdot_factor,
                    "k_ab_factor": params.k_ab_factor,
                    "dt0_mm": params.dt0_mm,
                })


def save_metadata_json(
    trajectories: list[tuple[NozzleConditionParams, dict]],
    output_path: str | Path,
    split_assignment: dict[str, str] | None = None,
) -> None:
    """Save per-trajectory metadata (params, n_steps, reached_failure, life_s)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records_out = []
    for params, rec in trajectories:
        split = (split_assignment or {}).get(params.trajectory_id, "train")
        records_out.append({
            "trajectory_id": params.trajectory_id,
            "condition_id": params.condition_id,
            "split": split,
            "reached_failure": bool(rec["reached_failure"]),
            "life_s": float(rec["time_s"][-1]),
            "n_steps": int(rec["n_steps"]),
            **{k: float(v) for k, v in asdict(params).items()
               if k not in ("trajectory_id", "condition_id")},
        })
    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "schema": "nozzle_sim_ode_v1",
            "k_ab_calibrated": _K_AB,
            "tau_th": _TAU_TH,
            "n_th": _N_TH,
            "h_ref": _H_REF,
            "trajectories": records_out,
        }, f, indent=2, ensure_ascii=False)
