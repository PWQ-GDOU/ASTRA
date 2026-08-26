# Strict Nozzle RUL Benchmark

- Created: `2026-07-24T12:17:22.588940+00:00`
- Frozen config hash: `767e1a8fe7299d4147e55bd6fe2f6bfd190ad29c0e15832e0433ec10c4ace7f9`
- Dataset SHA256: `2d1b4d906b8832e84f802eea449eb1ec0d744911604bd37cbfa70241e807a7c2`
- Execution mode: `canonical`
- Automatic claim audit: **The preregistered claim criteria are not all satisfied; no task-specific best claim is made.**
- Test policy: validation-only early stopping; test predicted once per locked run; no test-fitted seed, family, or ensemble weights.

## Rolling-fold macro metrics

| Method | Type | RMSE (s) | MAE (s) | Weighted RMSE (s) | NRMSE | Ranking |
|---|---|---:|---:|---:|---:|---|
| lstm | trainable | 6.030874 | 5.899512 | 6.011983 | 0.301544 | eligible |
| tcn | trainable | 6.182371 | 5.974916 | 6.186056 | 0.309119 | eligible |
| transformer | trainable | 6.408808 | 6.266181 | 6.358959 | 0.320440 | eligible |
| physics_input | trainable | 0.896667 | 0.885098 | 0.943340 | 0.044833 | eligible |
| physics_full | trainable | 0.949854 | 0.934497 | 0.964221 | 0.047493 | eligible |
| current_rate | baseline | 0.876673 | 0.835112 | 0.794136 | 0.043834 | eligible |
| local_slope | baseline | 0.997446 | 0.968159 | 0.927316 | 0.049872 | eligible |
| ridge | baseline | 0.798783 | 0.758948 | 0.814020 | 0.039939 | eligible |
| huber | baseline | 0.806856 | 0.766672 | 0.822000 | 0.040343 | eligible |
| clock_oracle | baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | diagnostic/excluded |

## Locked fold3 metrics

| Method | RMSE (s) | MAE (s) | Weighted RMSE (s) |
|---|---:|---:|---:|
| lstm | 11.016428 | 10.886328 | 10.986250 |
| tcn | 10.388646 | 10.243713 | 10.355965 |
| transformer | 10.874523 | 10.730885 | 10.842650 |
| physics_input | 0.052706 | 0.044480 | 0.051630 |
| physics_full | 0.052299 | 0.044061 | 0.051178 |
| current_rate | 0.017791 | 0.012432 | 0.015859 |
| local_slope | 0.027758 | 0.020422 | 0.025191 |
| ridge | 1.156260 | 1.140891 | 1.152085 |
| huber | 1.180478 | 1.164062 | 1.176027 |
| clock_oracle | 0.000000 | 0.000000 | 0.000000 |

## Ablations (rolling-fold macro)

| Dimension | Variant | Model | Window | RMSE (s) | Status |
|---|---|---|---:|---:|---|
| feature_tier | thermal_only | tcn | 3 | 7.068008 | predeclared |
| feature_tier | state_only | tcn | 3 | 7.194171 | predeclared |
| feature_tier | state_aware | tcn | 3 | 6.182371 | predeclared |
| feature_tier | age_aware | tcn | 3 | 6.034892 | diagnostic/excluded |
| window_size | 1 | tcn | 1 | 6.084750 | predeclared |
| window_size | 3 | tcn | 3 | 6.182371 | predeclared |
| window_size | 5 | tcn | 5 | 6.972634 | predeclared |
| physics | tcn | tcn | 3 | 6.182371 | predeclared |
| physics | physics_input | physics_input | 3 | 0.896667 | predeclared |
| physics | physics_full | physics_full | 3 | 0.949854 | predeclared |

The feature/window ablations use the predeclared TCN control so the external physics-RUL branch does not bypass a feature tier. The physics ablation is TCN vs physics-input vs full-physics at state-aware/window-3.

## Claim boundary

The preregistered claim criteria are not all satisfied; no task-specific best claim is made.
This single simulated trajectory cannot support a global SOTA claim or external-generalization claim.
Age-aware inputs and the exact `20-t` clock oracle are diagnostic and excluded from ranking.

## External same-domain status

**N/A.** No verified machine-readable public same-task nozzle RUL dataset was identified. See `docs/nozzle_public_data_audit.md`.

## Historical non-strict numbers (separate; not comparable)

Legacy values `2.501 / 2.674 / 3.819` are retained only as historical non-strict context. They are not substituted for any failed run and are not included in strict rankings or claims.

## Artifacts

- `frozen_config.json`
- `NOZZLE_STRICT_REPORT.json`
- `NOZZLE_STRICT_REPORT.md`
- `benchmark_summary.csv`
- `ablation_summary.csv`
- `predictions.csv`
- `environment.json`
- `per_seed_predictions/`
- `benchmark_rmse.png`
- `locked_fold3_predictions.png`
- `ablation_rmse.png`
