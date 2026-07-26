# COMSOL Reaction-Wheel Simulation Audit

Archive: `反作用轮全过程comsol仿真退化数据集.zip`

SHA256:

`b628deb89634cf8732cd96bb62d675b820aedb0aa337fd36082db53bfebca8fe`

## What The Archive Contains

### Aligned cycle dataset

- 5 complete trajectories.
- 300 aligned cycles per trajectory.
- Tracks 1, 2, 3 are training; track 4 is validation; track 5 is test.
- Each row contains physical cycle/hour RUL, RUL ratio, degradation indicators, operating telemetry, estimated physical parameters, failure threshold, and metadata.
- The metadata failure cycles are 269, 268, 257, 271, and 250 for tracks 1 through 5.
- RUL decreases exactly one cycle per aligned cycle and becomes zero at the declared failure cycle; post-failure rows are retained with zero RUL.

This is suitable for a simulation-domain RUL experiment. It is not the same task as FEMTO ordinal run-to-end RUL and must be reported separately.

### Low-error window dataset

- Four fault modes: bearing wear, inter-turn short circuit, permanent-magnet demagnetization, and combined degradation.
- Five trajectories per fault mode and 100 degradation levels.
- Preassigned trajectory split: tracks 1-3 train, track 4 validation, track 5 test.
- Seven windows per degradation level, giving 8,400 train rows, 2,800 validation rows, and 2,800 test rows.
- Train/validation/test sample IDs and source files are disjoint.
- RUL is provided as degradation level and life fraction.

## Feature Tiers

The adapter in `src/data/reaction_wheel_sim.py` exposes explicit tiers:

- `operational` / `noisy`: sensor-like telemetry and noisy window features.
- `estimated` / `observable`: operational telemetry plus estimated operational health quantities.
- `oracle`: true simulated degradation parameters and/or precomputed RUL estimates.

Oracle columns include true degradation state, true physical parameters, and particle-filter RUL estimates. They are audit-only and must not be used for the primary model result.

## Leakage-Safe Ridge Audit

The audit runner is:

```text
scripts/audit_reaction_wheel_sim.py
```

Output:

```text
outputs/reaction_wheel_sim_audit/SIMULATION_AUDIT.json
outputs/reaction_wheel_sim_audit/SIMULATION_RIDGE_RESULTS.csv
```

### Aligned data, fixed predeclared split

| Tier | Validation RMSE | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Operational | 4.382 | **7.625** | 6.369 |
| Estimated | 7.452 | 10.421 | 9.964 |
| Oracle audit-only | 8.848 | 10.617 | 10.272 |

The operational tier is the cleanest current simulation-domain baseline. The oracle tier is not a valid primary result because its inputs include simulated latent state and/or derived RUL estimates.

### Low-error window data, fixed trajectory split

| Tier | Validation RMSE | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Noisy | 10.843 | **8.162** | 6.784 |
| Observable | 11.032 | 8.426 | 7.083 |
| Oracle audit-only | 0.0007 | 0.0014 | 0.0014 |

The near-zero oracle result confirms that the generated table contains highly direct target-related variables. It must never be used to advertise model performance.

## Protocol Boundary

The supplied split is trajectory-disjoint and therefore usable as a predeclared simulation benchmark. However, there are only five base trajectories and all four fault modes reuse the same trajectory IDs. The primary result should therefore report:

- fixed split test performance on track 5;
- fault-mode breakdown;
- train/validation/test trajectory IDs;
- operational versus oracle feature tiers;
- raw cycle/level RMSE and MAE;
- no aggregation with FEMTO bearing proxy results.

The simulation result can support the claim that the algorithm works on the supplied COMSOL reaction-wheel simulation domain. It does not establish real spacecraft reaction-wheel performance or universal reaction-wheel SOTA.
