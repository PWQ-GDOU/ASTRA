"""One-shot patch: add physics_observable tier to nozzle_multitrajectory.py"""
import ast, pathlib, sys

path = pathlib.Path('/data/disk3/spacecraft_rul/src/data/nozzle_multitrajectory.py')
src  = path.read_text(encoding='utf-8')

# ── 1. Add PHYSICS_OBSERVABLE_FEATURES constant ──────────────────────────
OLD1 = (
    'ESTIMATED_FEATURES = OBSERVABLE_FEATURES + (\n'
    '    "cumulative_ablation_depth_mm",\n'
    '    "ablation_rate_m_s",\n'
    ')'
)
NEW1 = OLD1 + '''

# Physics-derived observable features: cumulative depth proxy from
# integrated Q*P^n, and current rate proxy.  Both are computable purely
# from in-situ sensor measurements (no privileged simulation state).
_K_AB_NOM: float = 1.7418e-11   # m3/J  calibrated on COMSOL 20-s reference
_P_REF_PA: float = 1.43e6       # Pa    reference chamber pressure
_N_PRES: float   = 0.40         # pressure exponent (oxidative ablation law)
PHYSICS_OBSERVABLE_FEATURES = OBSERVABLE_FEATURES + (
    "depth_proxy_mm",    # K_ab * integral(Q * (P/P_ref)^n * dt)  [mm]
    "rate_proxy_m_s",    # K_ab * Q_cur * (P_cur/P_ref)^n          [m/s]
)'''

assert OLD1 in src, 'Pattern 1 not found'
src = src.replace(OLD1, NEW1, 1)

# ── 2. Add physics_observable branch in tier if-elif ─────────────────────
OLD2 = (
    '    if feature_tier == "observable":\n'
    '        feature_names = OBSERVABLE_FEATURES\n'
    '    elif feature_tier == "estimated":\n'
    '        feature_names = ESTIMATED_FEATURES\n'
    '    else:\n'
    '        raise ValueError("feature_tier must be \'observable\' or \'estimated\'; oracle columns are forbidden")'
)
NEW2 = (
    '    if feature_tier == "observable":\n'
    '        feature_names = OBSERVABLE_FEATURES\n'
    '    elif feature_tier == "estimated":\n'
    '        feature_names = ESTIMATED_FEATURES\n'
    '    elif feature_tier == "physics_observable":\n'
    '        feature_names = PHYSICS_OBSERVABLE_FEATURES\n'
    '    else:\n'
    '        raise ValueError(\n'
    '            "feature_tier must be \'observable\', \'estimated\', or "\n'
    '            "\'physics_observable\'; oracle columns are forbidden"\n'
    '        )'
)
assert OLD2 in src, 'Pattern 2 not found'
src = src.replace(OLD2, NEW2, 1)

# ── 3. Replace feature loading block ─────────────────────────────────────
OLD3 = '        features = np.column_stack([[_number(row, name) for row in ordered] for name in feature_names]).astype(np.float64)'
NEW3 = (
    '        if feature_tier == "physics_observable":\n'
    '            obs_raw = np.column_stack(\n'
    '                [[_number(row, n) for row in ordered] for n in OBSERVABLE_FEATURES]\n'
    '            ).astype(np.float64)\n'
    '            heat_flux = obs_raw[:, 1]   # heat_flux_W_m2\n'
    '            pressure  = obs_raw[:, 2]   # pressure_Pa\n'
    '            q_scaled  = heat_flux * (pressure / _P_REF_PA) ** _N_PRES\n'
    '            dt_arr    = np.diff(time_s, prepend=time_s[0])\n'
    '            dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 1.0\n'
    '            depth_proxy_m  = _K_AB_NOM * np.cumsum(q_scaled * dt_arr)\n'
    '            rate_proxy_m_s = _K_AB_NOM * q_scaled\n'
    '            features = np.column_stack([\n'
    '                obs_raw,\n'
    '                depth_proxy_m * 1000.0,\n'
    '                rate_proxy_m_s,\n'
    '            ]).astype(np.float64)\n'
    '        else:\n'
    '            features = np.column_stack(\n'
    '                [[_number(row, name) for row in ordered] for name in feature_names]\n'
    '            ).astype(np.float64)'
)
assert OLD3 in src, 'Pattern 3 not found'
src = src.replace(OLD3, NEW3, 1)

# ── Syntax check ─────────────────────────────────────────────────────────
ast.parse(src)
path.write_text(src, encoding='utf-8')
print('Patch applied and syntax OK')
print(f'PHYSICS_OBSERVABLE_FEATURES in source: {("PHYSICS_OBSERVABLE_FEATURES" in src)}')
