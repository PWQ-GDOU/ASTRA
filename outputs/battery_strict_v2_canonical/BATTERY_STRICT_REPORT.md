# Strict Battery Benchmark

- Protocol: nested cell-level LOOCV; strict14 exact labels.
- Exact-event macro covers B0005/B0006/B0018 only; B0007 strict14 is right-censored.
- `selected_neural` selects a neural candidate by inner neural RMSE.
- `selected_method_blend` separately selects candidate and Ridge blend alpha by inner blend RMSE; alpha=0 is an explicit Ridge fallback.
- Selection and final use the same seeds; final retraining keeps the frozen scheduler budget of 160 epochs.
- Test labels are used only by the final evaluator.
- Nozzle results are frozen and untouched.

## Exact-Event Macro Mean

| Method | RMSE | MAE | Bias | NRMSE / 100 cycles |
|---|---:|---:|---:|---:|
| selected_neural | 12.0237 | 10.2893 | 5.4118 | 0.1202 |
| selected_method_blend | 12.7966 | 11.0869 | 5.6110 | 0.1280 |
| ridge | 13.8979 | 11.9477 | 0.7180 | 0.1390 |
| huber | 22.5308 | 21.8444 | -4.3463 | 0.2253 |
| soh_monotone | 15.2230 | 12.7003 | 0.3643 | 0.1522 |
| capacity_trend | 27.2908 | 20.3012 | 3.7908 | 0.2729 |

## Fixed Neural Families

| Family | RMSE | MAE | Bias |
|---|---:|---:|---:|
| ms_l16 | 10.8538 | 9.1640 | 5.2117 |
| life_l16 | 11.1371 | 9.5854 | 5.8014 |
| gru_l16 | 13.0690 | 11.1018 | 6.4680 |
| transformer_l16 | 11.2360 | 9.4503 | 5.5696 |

## Outer Folds

| Holdout | Status | Neural candidate | Method candidate | Alpha | Ridge fallback | Neural RMSE | Method RMSE |
|---|---|---|---|---:|---|---:|---:|
| B0005 | event_observed | life_l24 | life_l24 | 0.94 | no | 7.5557 | 7.0519 |
| B0006 | event_observed | transformer_l16 | gru_l16 | 0.59 | no | 9.5045 | 12.3272 |
| B0007 | right_censored | life_l24 | life_l24 | 0.95 | no | N/A | N/A |
| B0018 | event_observed | transformer_l16 | transformer_l16 | 1.00 | no | 19.0108 | 19.0108 |

## B0007 Adaptive Special

- EOL: 1.504530 Ah
- Life cycle: 124.0
- Neural candidate: `life_l24`
- Method candidate: `life_l24`; alpha=0.95
- Neural RMSE: 8.2920 cycles
- Method RMSE: 8.1662 cycles
- Ridge RMSE: 7.9274 cycles

This adaptive row has a separate EOL-derived capacity-margin feature distribution and is not mixed with strict14.
