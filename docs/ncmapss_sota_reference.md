# N-CMAPSS SOTA Reference Table

Literature results for the N-CMAPSS turbofan engine degradation benchmark.
These numbers are from published papers and may use slightly different protocols.
Our results must be compared with matched protocol (same dataset subset, RUL cap,
train/test split, and metric definition).

## Protocol conventions in this project

- Dataset: DS01–DS08 (separate evaluations)
- RUL cap: 125 cycles
- Split: provided unit-disjoint dev/test (no random re-splitting)
- Validation: last 20% of dev units for checkpoint selection only
- Seeds: 42, 123, 456, 2026, 3407 — equal-weight ensemble
- Metric: RMSE (cycles), MAE, NASA asymmetric score
- No test labels in any selection step

## Reference results from literature

### DS01–DS02 (complete degradation trajectories)

| Method | DS01 RMSE | DS02 RMSE | Year | Notes |
|---|---:|---:|---|---|
| LSTM baseline | ~12–15 | ~14–18 | 2021 | Chao et al. |
| CNN-LSTM | ~9–12 | ~12–16 | 2022 | Various |
| Transformer | ~8–11 | ~10–14 | 2022–2023 | Multiple papers |
| BiLSTM-Attention | ~8–10 | ~11–14 | 2022 | |
| Physics-informed NN | ~7–10 | ~9–13 | 2023 | |
| SOTA (2023–2024) | **~6–8** | **~8–11** | 2024 | Best published results |

### DS03–DS04 (truncated trajectories, harder)

| Method | DS03 RMSE | DS04 RMSE | Year |
|---|---:|---:|---|
| LSTM | ~13–18 | ~15–20 | 2021 |
| Transformer | ~10–14 | ~12–16 | 2022–2023 |
| SOTA (2024) | **~8–12** | **~10–14** | 2024 |

## Important caveats for comparison

1. Many papers only evaluate on DS01 or report selected subsets.
2. Some papers use 5-fold cross-validation within dev units rather than
   the provided unit-disjoint split — not directly comparable.
3. RUL cap conventions differ (some use 130 or no cap).
4. NASA asymmetric score is rarely reported consistently.
5. Operational condition normalisation details vary significantly.

## Our SOTA strategy

The condition-aware design (ConditionNormLayer FILM) specifically targets
the variable flight condition challenge in N-CMAPSS, which is the main
source of variance across methods. Expected advantages:

- ConditionAwareMultiScaleTCN: multi-scale temporal + condition de-bias
- BiGRUAttention: long-range dependencies + temporal importance weighting
- TransformerRUL: global attention + auxiliary degradation head
- PhysicsInformedGRU: physical trend prior + learned residual

Target: DS01 RMSE ≤ 8.0 cycles, DS02 RMSE ≤ 11.0 cycles with 5-seed ensemble.

## Data download

```
https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
```

Extract DS01.h5–DS08.h5 into `data/processed/ncmapss/`.

## Run command

```bash
python scripts/exp_ncmapss_strict.py \
    --data-dir data/processed/ncmapss \
    --datasets DS01 DS02 DS03 DS04 \
    --output outputs/ncmapss_strict_v1 \
    --device cuda:0
```
