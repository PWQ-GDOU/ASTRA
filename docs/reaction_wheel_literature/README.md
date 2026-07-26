# Reaction-Wheel Proxy Literature Checklist

This project currently evaluates FEMTO/PRONOSTIA bearing Learning_set runs as a mechanical proxy. These papers must not be treated as directly comparable unless their labels, forecast origin, split, preprocessing, and metric units are reproduced locally.

## Priority A: download first

1. von Hahn and Mechefske (2022), knowledge-informed Weibull loss for bearing RUL.
   - DOI: 10.22215/jphm.v2i1.3162
   - arXiv: https://arxiv.org/abs/2201.01769
   - Code candidate: https://github.com/tvhahn/weibull-knowledge-informed-ml
   - Reproduction value: loss and monotonic degradation constraint.

2. 2025 PRONOSTIA comparative study.
   - DOI: 10.1109/ACCESS.2025.3551772
   - Reproduction value: recent protocol and baseline audit. Do not copy headline numbers without matching the protocol.

3. Yang et al. (2022), regression shapelet and graph neural network for RUL.
   - DOI: 10.1109/TIM.2022.3151169
   - Reproduction value: structured temporal/degradation representation if code or supplementary material is available.

## Priority B

4. Lu et al. (2021), deep adversarial learning prognostics.
   - DOI: 10.1109/TAI.2021.3097311

5. Multi-domain TCN (2024).
   - DOI: 10.3390/app14062354

6. Contrastive self-supervised RUL (2023).
   - DOI: 10.1016/j.ifacol.2023.10.604

7. Wang et al. (2022), LSTM with uncertainty quantification.
   - DOI: 10.3390/s22124549
   - Use only when uncertainty metrics are added to the strict benchmark.

## Required audit fields for every downloaded paper

- PDF filename and SHA256.
- Dataset split and bearing IDs.
- RUL/EOL label definition.
- Forecast origin and sequence length.
- Feature preprocessing and normalization denominator.
- Seed count and checkpoint-selection rule.
- Raw metric units and aggregation rule.
- Public code or supplementary URL.
- Directly comparable: yes/no, with reason.

## Current protocol boundary

The strict proxy benchmark uses six Learning_set bearings, end-of-run ordinal labels, bearing-level nested leave-one-bearing-out validation, train-bearing-only scaling, five seeds, raw RMSE/MAE/bias, train-derived normalized RMSE, and descriptive test-derived compatibility metrics. It is not a real spacecraft reaction-wheel benchmark.
