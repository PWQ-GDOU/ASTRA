# ASTRA Final Results Table

更新时间：2026-08-19  
主实验：`cross_component_transfer_v3`  
远程运行状态：`78/78` cells completed，`0` failures

## 结论摘要

| 链路 | 设置 | Transfer RMSE | Scratch RMSE | 相对提升 | 胜出 outer folds | 正向证据 |
|---|---:|---:|---:|---:|---:|---|
| 喷嘴 → NASA 电池 | N=1 | 51.119 | 50.098 | -2.04% | 1/3 | 否，diagnostic |
| 喷嘴 → NASA 电池 | N=2 | 46.213 | 47.128 | +1.94% | 1/3 | 否，diagnostic |
| 喷嘴 → FEMTO 机械退化代理 | N=1 | 883.687 | 988.699 | +10.62% | 5/6 | 是 |
| 喷嘴 → FEMTO 机械退化代理 | N=2 | 768.201 | 835.456 | +8.05% | 5/6 | 是 |
| 喷嘴 → FEMTO 机械退化代理 | N=3 | 1212.669 | 1193.792 | -1.58% | 2/6 | 否，diagnostic |

接收规则是：相对 scratch 的 macro RMSE 至少改善 5%，且多数 outer folds 胜出。电池和 FEMTO 代理分别判定，不能用代理链路替代真实电池或真实反作用轮证据。

## Macro 指标

原始标签单位保持为各目标域的剩余周期/剩余测量数。`bias = mean(prediction - truth)`；`normalized_rmse` 为协议中按训练侧标签尺度计算的归一化 RMSE。

| 域 | N | 方法 | Raw RMSE | MAE | Bias | Normalized RMSE | Outer folds | 胜 scratch | 相对提升 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Battery | 1 | transfer | 51.119 | 44.567 | 42.063 | 0.540 | 3 | 1/3 | -2.04% |
| Battery | 1 | scratch | 50.098 | 43.418 | 41.186 | 0.531 | 3 | — | — |
| Battery | 1 | Ridge | 53.699 | 47.082 | 45.301 | 0.569 | 3 | — | — |
| Battery | 2 | transfer | 46.213 | 39.250 | 35.874 | 0.450 | 3 | 1/3 | +1.94% |
| Battery | 2 | scratch | 47.128 | 40.226 | 37.870 | 0.456 | 3 | — | — |
| Battery | 2 | Ridge | 49.600 | 42.854 | 40.172 | 0.478 | 3 | — | — |
| FEMTO proxy | 1 | transfer | 883.687 | 831.171 | 565.332 | 0.570 | 6 | 5/6 | +10.62% |
| FEMTO proxy | 1 | scratch | 988.699 | 935.274 | 690.226 | 0.618 | 6 | — | — |
| FEMTO proxy | 1 | Ridge | 1142.229 | 1086.798 | 880.220 | 0.696 | 6 | — | — |
| FEMTO proxy | 2 | transfer | 768.201 | 705.641 | 404.594 | 0.532 | 6 | 5/6 | +8.05% |
| FEMTO proxy | 2 | scratch | 835.456 | 774.655 | 477.721 | 0.570 | 6 | — | — |
| FEMTO proxy | 2 | Ridge | 962.837 | 901.215 | 642.450 | 0.633 | 6 | — | — |
| FEMTO proxy | 3 | transfer | 1212.669 | 1150.557 | 871.712 | 0.623 | 6 | 2/6 | -1.58% |
| FEMTO proxy | 3 | scratch | 1193.792 | 1134.559 | 867.263 | 0.619 | 6 | — | — |
| FEMTO proxy | 3 | Ridge | 1251.091 | 1168.973 | 909.691 | 0.640 | 6 | — | — |

## Outer-Fold 指标

### NASA strict14 电池

| N | Holdout | Transfer RMSE / MAE / Bias | Scratch RMSE / MAE / Bias | Transfer 胜出 |
|---:|---|---:|---:|---|
| 1 | B0005 | 43.077 / 35.381 / 28.976 | 43.153 / 35.295 / 29.591 | 是 |
| 1 | B0006 | 57.355 / 51.041 / 49.932 | 54.400 / 47.734 / 46.740 | 否 |
| 1 | B0018 | 52.925 / 47.280 / 47.280 | 52.742 / 47.227 / 47.227 | 否 |
| 2 | B0005 | 49.124 / 41.187 / 33.004 | 45.708 / 37.561 / 32.755 | 否 |
| 2 | B0006 | 40.790 / 33.013 / 31.073 | 39.579 / 31.840 / 29.580 | 否 |
| 2 | B0018 | 48.724 / 43.550 / 43.550 | 56.089 / 51.271 / 51.271 | 是 |

### FEMTO 六轴承 outer LOO

| N | Holdout | Transfer RMSE | Scratch RMSE | Transfer 胜出 |
|---:|---|---:|---:|---|
| 1 | Bearing1_1 | 858.69 | 894.61 | 是 |
| 1 | Bearing1_2 | 1038.40 | 1080.18 | 是 |
| 1 | Bearing2_1 | 589.57 | 816.61 | 是 |
| 1 | Bearing2_2 | 957.35 | 1276.58 | 是 |
| 1 | Bearing3_1 | 1036.80 | 1027.87 | 否 |
| 1 | Bearing3_2 | 821.31 | 836.33 | 是 |
| 2 | Bearing1_1 | 856.33 | 902.13 | 是 |
| 2 | Bearing1_2 | 505.85 | 506.36 | 是 |
| 2 | Bearing2_1 | 750.87 | 1056.35 | 是 |
| 2 | Bearing2_2 | 871.77 | 952.67 | 是 |
| 2 | Bearing3_1 | 1125.69 | 1085.97 | 否 |
| 2 | Bearing3_2 | 498.69 | 509.27 | 是 |
| 3 | Bearing1_1 | 984.99 | 996.63 | 是 |
| 3 | Bearing1_2 | 931.16 | 913.87 | 否 |
| 3 | Bearing2_1 | 1431.40 | 1568.45 | 是 |
| 3 | Bearing2_2 | 1362.63 | 1268.82 | 否 |
| 3 | Bearing3_1 | 1399.08 | 1360.70 | 否 |
| 3 | Bearing3_2 | 1166.75 | 1054.29 | 否 |

## 协议与审计边界

- 源域是严格 200 条多轨迹喷嘴数据；输入为 `health = 1 - normalized_depth`，按 trajectory 划分 train/validation/source-test。
- 电池只把 B0005、B0006、B0018 作为 event-observed outer folds；B0007 是 right-censored，30 条诊断记录不计入精确 RMSE macro。
- FEMTO 只作为机械退化代理；每一 fold 的 scaler/adapter 只使用训练轴承，RMS 校准只使用 fine-tune 早期 prefix，禁止 holdout 和 full-life min/max。
- transfer、scratch、frozen encoder 使用相同 target architecture、epoch budget、chronological validation、early stopping 和 5 个固定 seeds。
- 主 transfer 是 pretrained encoder + target projection + target head 全量微调；frozen encoder 仅为正式消融。
- 所有产物均在 `outputs/cross_component_transfer_v3/`，包括 `PROTOCOL.json`、`DATA_MANIFEST.json`、`SUPERVISOR_STATUS.json`、报告、宏观表和逐样本预测。

## 赛事评分依据与当前状态

仓库和交接材料中没有发现可核验的官方赛事评分规则文件，因此不能把内部估算写成正式成绩。此前约 `79/100` 只是非官方技术部分估算；v3 完成后应更新技术证据描述，但不应擅自把它改成官方分数。

当前可以诚实计入的技术证据是：严格协议、可追溯数据与切分、5-seed 对照、NASA 电池的右删失处理，以及 FEMTO 代理 N=1/2 的正向迁移证据。电池 v3、FEMTO N=3、真实反作用轮验证和官方赛事总分仍分别保持 diagnostic/未验证状态。

## Artifact

- `outputs/cross_component_transfer_v3/CROSS_TRANSFER_V3_REPORT.json`
- `outputs/cross_component_transfer_v3/CROSS_TRANSFER_V3_MACRO.csv`
- `outputs/cross_component_transfer_v3/CROSS_TRANSFER_V3_PREDICTIONS.csv`
- `outputs/cross_component_transfer_v3/PROTOCOL.json`
- `outputs/cross_component_transfer_v3/DATA_MANIFEST.json`
- `outputs/cross_component_transfer_v3/SUPERVISOR_STATUS.json`
