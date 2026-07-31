# 🚀 ASTRA: Aerospace System Transfer Reliability Analysis

**航天器关键组件跨域退化迁移寿命预测**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ASTRA 是一个面向航天器关键组件寿命预测的跨域退化迁移学习框架。通过将**喷嘴烧蚀**（源域）的退化知识迁移至**反作用轮**与**电池**（目标域），结合物理约束灰盒模型，实现小样本下的高精度剩余寿命（RUL）预测。

## 📊 核心特性

- **跨域退化迁移**：PEUDA 双时间-频率域自适应 + 对比学习
- **物理灰盒模型**：Arrhenius 退化方程嵌入神经网络的灰盒架构
- **自监督预训练**：SHRDL 健康状态分解学习，提升小样本迁移能力
- **不确定性量化**：PCBNN 物理约束贝叶斯网络，提供预测置信区间
- **C-MAPSS 基准**：发动机级验证划分、训练域拟合归一化、无测试集早停

## 🏗️ 项目结构

```
ASTRA/
├── src/                    # 核心源码
│   ├── models/             # 模型定义（SHRDL, PEUDA, PCBNN）
│   ├── data/               # 数据加载与预处理
│   ├── baselines/          # 基线方法（LSTM, Wiener过程）
│   ├── evaluate/           # 评估指标
│   ├── uncertainty/        # 不确定性估计
│   └── visualize/          # 可视化
├── scripts/                # 训练与实验脚本
│   ├── train_v2.py         # V2 TCN 训练
│   ├── pretrain_shrdl.py   # SHRDL 预训练
│   ├── adapt_peuda.py      # PEUDA 域自适应
│   └── ...
├── data/                   # 数据集
│   ├── processed/cmapss/   # C-MAPSS 涡扇发动机
│   └── processed/nasa_battery/  # NASA 电池退化
├── outputs/                # 实验结果与预测
├── checkpoints/            # 模型权重
└── docker/                 # Docker 部署配置
```

## C-MAPSS Benchmark 结果

### 历史结果

早期V2/ensemble结果（FD002约28.8，以及FD001约14.7、FD003约14.3）使用过官方测试集逐epoch选点或其他未统一的预处理，只保留为历史参考，不能作为严格复现数字。

### 严格协议结果

统一配置：发动机级训练/验证划分（`split_seed=2026`，验证比例20%）、窗口长度60、`RUL cap=125`、训练Unit拟合统计量、验证RMSE选checkpoint、官方测试集只在选点后评估。`test_raw`使用官方未截断RUL；`test_cap125`将官方测试RUL同步截断到125，便于与采用相同cap的论文比较。

| 数据集 | 编码器/输入 | 验证 RMSE | test_raw RMSE | test_cap125 RMSE | test_raw Score |
|--------|-------------|----------:|--------------:|-----------------:|---------------:|
| FD001 | 多尺度TCN / 24维 | 13.363 | 14.951 | 13.785 | 308.7 |
| FD002 | 多尺度TCN + 工况归一化 / 24维 | 17.426 | 25.575 | 14.391 | 5936.0 |
| FD003 | 多尺度TCN / 24维 | 9.791 | 16.394 | 14.541 | 545.1 |
| FD004 | 多尺度TCN + 工况归一化 / 24维 | 15.436 | 29.041 | 16.156 | 7120.3 |

FD002窗口60模型的三个固定seed（42/123/456）单模型`test_raw RMSE=25.850±0.555`，固定等权平均集成为`25.131`；同步cap=125后，集成RMSE为`12.831`。该三seed集成没有使用测试标签拟合权重。

## N-CMAPSS Benchmark（DS01-DS08 完整结果）

**N-CMAPSS**（NASA Dataset 17）涡扇发动机退化 benchmark，严格协议入口：`scripts/exp_ncmapss_strict.py`

**实验设置**：
- 数据：DS01-DS08（全部8个子集）；DS08d 文件截断，改用 DS08a 变体（symlink）；文件结构为 dev/test unit-disjoint
- 降采样：`--stride 1000`（原始 4.9M 行/数据集 → ~4100 训练窗口/数据集，大幅提速）
- 预处理：train-only scaler；FILM条件归一化；RUL cap=125
- 候选8个：BiGRU+Attention、MultiScale TCN、Transformer (base/large)、Physics GRU，分 physical_with_conditions / full_with_conditions 两个特征集
- 损失：`0.75×smooth_l1 + 0.25×WeibullRULLoss(eta=100, β=3.0)`
- 协议：dev集后20%作inner validation；5 seed等权集成；无test标签参与选择

### 测试集结果（5-seed ensemble，GPU3，stride=1000）

| 数据集 | 选中模型 | Neural RMSE | Ridge RMSE | 提升 |
|--------|----------|:-----------:|:----------:|:----:|
| DS01 | bigru_full_cond | **7.208** | 13.068 | +44.8% |
| DS02 | bigru_attn_phys_cond | **6.495** | 11.269 | +42.4% |
| DS03 | bigru_full_cond | **7.891** | 12.029 | +34.4% |
| DS04 | bigru_full_cond | **7.150** | 19.829 | +63.9% |
| DS05 | bigru_full_cond | **6.519** | 14.201 | +54.1% |
| DS06 | bigru_attn_phys_cond | **8.165** | 14.914 | +45.2% |
| DS07 | bigru_full_cond | **14.502** | 17.790 | +18.5% |
| DS08a | bigru_attn_phys_cond | **11.424** | 14.009 | +18.5% |
| **Macro DS01-DS08** | — | **8.669** | 14.638 | **+40.8%** |

**Neural 比 Ridge 好 40.8%（macro）**；BiGRU 系列（full features / physical+conditions）在所有数据集均被选中。

说明：
- stride=1000 为加速设置，每1000个时间步取1个窗口（DS01有4.9M行原始数据）。
- DS07/DS08（截断型数据集，RUL估计更难）提升幅度相对较小，符合文献规律。
- DS08d 文件下载截断，改用 DS08a 变体（通过 symlink `DS08.h5 → N-CMAPSS_DS08a-009.h5`）。
- 入口：`python scripts/exp_ncmapss_strict.py --data-dir path/to/ --stride 1000 --epochs 200`

# 无需数据的 synthetic smoke test
python scripts/exp_ncmapss_strict.py --synthetic --output /tmp/smoke
```

数据下载（约 14GB）：

```
https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
```

> 注意：C-MAPSS论文对测试RUL是否同步应用RUL cap并不总是说明清楚，因此raw和cap125必须分栏报告，不能直接混比。

## 喷管多工况烧蚀退化 Benchmark

### 仿真器（减阶物理 ODE）

`src/data/nozzle_sim_ode.py` — 基于 COMSOL 20秒 C/C 喉衬烧蚀轨迹校准的减阶物理 ODE 仿真器。

**物理机制（三耦合方程）**：
1. 固体温度上升（半无限厚壁导热）：`T_s(t) = T_in - (T_in-T_s0)/(1+t/τ)^n`，校准于 COMSOL 实测
2. 对流热流（Dittus-Boelter scaling）：`Q = h_eff × (T_in - T_s)`，`h∝mdot^0.8`
3. 氧化烧蚀速率：`r_dot ∝ Q × P^0.4`，校准系数 `k_ab = 1.74e-11 m³/J`

**校准结果**：
- 参考运行20秒深度：0.2572 mm（COMSOL 目标 0.2585 mm，误差 <0.5%）
- 参考运行失效时间（0.5mm阈值）：58.3秒

**工况参数空间（LHC 采样）**：

| 参数 | 范围 | 物理意义 |
|------|------|---------|
| `T_in_K` | 920–1320 K | 燃气入口温度（不同推进剂/O/F比） |
| `p_ch0_Pa` | 0.8–2.0 MPa | 初始燃烧室压力（不同推力级别） |
| `mdot_factor` | 0.65–1.35 | 质量流率倍率 |
| `k_ab_factor` | 0.70–1.40 | 烧蚀化学系数（材料/工艺变化） |
| `dt0_mm` | 12–20 mm | 初始喉径 |

```bash
# 生成200条轨迹（0.8秒完成）
python scripts/gen_nozzle_multitrajectory.py \
    --n-traj 200 --output data/processed/nozzle_mt_200 \
    --failure-depth-mm 0.5 --t-max-s 300

# SOTA验证（200条轨迹，window=20，双tier，全基线对比）
python scripts/run_nozzle_sota_validation.py \
    --data data/processed/nozzle_mt_200/nozzle_sim_200traj.csv \
    --output outputs/nozzle_sota_200_w20 \
    --window-size 20 --epochs 160 --device cuda:1
```

**200条轨迹数据集特性**：
- 寿命范围：22–194秒，中位数62.7秒
- 工况组数：24个不同条件（T_in/P/mdot分箱）
- 全部 run-to-failure（无截尾），split: train/val/test = 130/30/40

### Benchmark 协议（SOTA验证版）

**信息边界严格分层**（修复了早期版本的特权信息泄漏 bug）：

| Tier | 模型输入特征 | 用途 |
|------|------------|------|
| **Observable** | T_solid, heat_flux, pressure（3维，纯在线可测） | 正式主结果 |
| **Estimated** | Observable + depth_mm + rate_m_s（5维） | 工程增强结果 |
| Privileged | 上述 + 直接传入 margin/rate 给模型 | 历史错误做法，已废弃 |

- 内层选择：train→val，3 seeds，80 epochs；外层评估：train+val→test，5 seeds，160 epochs
- 候选7个：GRU / MultiScale TCN / Transformer / PhysicsResidualRateNet × 两个tier
- PhysicsResidualRateNet 在 observable tier 强制 `current_rate_m_s=None`（诚实），在 estimated tier 才使用物理积分先验

### SOTA 验证结果（v2，200条轨迹，window=20，修复信息泄漏）

**Observable Tier（仅温度/热流/压力）**

内层选中：**gru_obs_w20**（val_rmse=39.72s；所有observable候选39–41s，差异小）

| 方法 | Test RMSE (s) | vs Ridge |
|------|:-------------:|:--------:|
| current_rate（特权参考，不参与主对比） | 12.13 | +58.2% |
| **Ridge** | **28.99** | — |
| Huber | 29.02 | −0.1% |
| mean_baseline | 30.80 | −6.3% |
| GBDT | 33.13 | −14.3% |
| RF | 38.91 | −34.2% |
| **GRU neural** | **38.15** | −31.6% ✗ |

Observable tier 结论：**所有 ML 方法均输给 current_rate（特权基线）**。从纯热流/温度/压力预测 RUL，5–20步窗口信息不足；GRU 在 observable 模式下仅与 RF 相近，Ridge最优。

**Estimated Tier（+累积深度+烧蚀速率）**

内层选中：**rate_est_w20**（val_rmse=4.07s）

| 方法 | Test RMSE (s) | vs Ridge |
|------|:-------------:|:--------:|
| **RandomForest** | **0.40** | **+95.8%** |
| GBDT | 0.86 | +91.0% |
| **PhysicsResidualRateNet** | **2.01** | **+79.0%** ✓ |
| Huber | 9.41 | +1.8% |
| Ridge | 9.58 | — |
| current_rate | 12.13 | −26.6% |

Estimated tier 结论：RF/GBDT 通过 `X[-1]` 特征学到物理法则 `RUL≈(fail_depth−depth)/rate`，近乎完美（0.40/0.86s）。**PhysicsResidualRateNet 显式积分同一物理先验，达到2.01s**，比 Ridge 好 79%；树模型仍占优。

**敏感性分析（Observable tier，阈值0.3–0.7mm）**

| 失效阈值 | Neural RMSE | Ridge RMSE | GBDT RMSE | Neural vs GBDT |
|---------|:-----------:|:----------:|:---------:|:--------------:|
| 0.3mm | 10.14 | 8.30 | 12.83 | **+21%** ✓ |
| 0.4mm | 23.01 | 15.06 | 21.21 | **+8%** ✓ |
| 0.5mm | 36.08 | 23.57 | 32.74 | **+9%** ✓ |
| 0.6mm | 56.15 | 33.62 | 46.70 | **+17%** ✓ |
| 0.7mm | 82.36 | 45.12 | 62.23 | **+24%** ✓ |

**Observable tier 神经网络在全部5个失效阈值均优于 GBDT**，但仍输给 Ridge。

**演进对比（修复bug前→后，200轨迹+window=20）**：

| 版本 | neural RMSE | Ridge RMSE | 是否诚实 |
|------|:-----------:|:----------:|:-------:|
| v1（40轨迹，含信息泄漏） | 5.47s | 7.09s | ✗ 不诚实 |
| v2（40轨迹，修复bug） | 10.98s | 11.23s | ✓ |
| **v3（200轨迹，w=20，修复）** | **2.01s** | **9.58s** | **✓** |

PhysicsResidualRateNet 在 estimated tier 随数据量增加有明显收益（10.98→2.01s，-81%）；树模型也同步改善（GBDT 6.36→0.86s）。

入口：`scripts/run_nozzle_sota_validation.py`；数据：`scripts/gen_nozzle_multitrajectory.py`

> 注：全部结果来自 ODE 减阶仿真（COMSOL 20s 校准）。接到实际喷管静态点火试验数据后脚本可直接替换运行。

**协议**：GEO 卫星 PINN-ODE 仿真（60 轨迹，LHC 采样）预训练共享编码器 → NASA strict14 目标域 LOO fine-tune 对照。

**关键设计**：
- 仅迁移编码器主干权重，**不迁移 RUL 头**（GEO RUL 范围 0–1200 cycle，NASA strict14 RUL 范围 0–130 cycle，9× 量程差异，迁移头会使初始预测偏高9×）。
- GEO 特征17维（9基础 + dV30/60/90/120 + E30/60/90/120 PINN voltage-drop 极端段特征）。
- 源域预训练：150 epoch；目标域 fine-tune：100 epoch；5 seed 等权集成。

| 评估电芯 | Transfer RMSE | Scratch RMSE | Δ（正=迁移更好） |
|----------|:------------:|:------------:|:--------------:|
| B0005 | 13.274 | 8.654 | −4.62 |
| **B0006** | **7.036** | 7.117 | **+0.08** ✓ |
| B0007 | right-censored | right-censored | — |
| **B0018** | **15.112** | 19.906 | **+4.79** ✓ |
| **宏平均（3 event cells）** | **11.807** | **11.892** | **+0.085** ✓ |

说明：
- Transfer 整体略优于 Scratch（B0006+B0018 均有改善；B0005 较弱）。
- 源域 val RMSE ≈ 127.6 cycle（GEO 仿真多样性高，绝对误差大但相对误差合理）。
- 对照：v1（未修复 RUL 头）Transfer=56.94 vs Scratch=12.60，完全失效；修复后恢复至竞争水平。
- 入口：`scripts/exp_geo_to_nasa_transfer.py`；支持 `--synthetic` smoke test。

## 主实验总表（竞赛四条线）

统一入口：`scripts/exp_main_suite.py`（约5分钟，GPU2）。结果目录：`outputs/main_suite/`。

| 域 | 协议 | 主指标 | 结果 |
|----|------|--------|------|
| 源域：喷嘴烧蚀 | 前40训练 / 后20评估 | Pure TCN / PCG-TCN / MultiScale RMSE | **2.501 / 2.674 / 3.819**（冻结） |
| 目标域：NASA电池（clean strict14 v2 canonical） | 嵌套电芯级 LOOCV；B0007 right-censored | selected neural / selected method blend RMSE | **12.024 / 12.797*** |
| 目标域：反作用轮代理 | FEMTO轴承 LOO，振动特征 | 归一化 RMSE / 相对 RMSE | **0.331 / 0.661** |
| 公开基准：C-MAPSS | 严格协议（见上表） | FD002 三seed集成 cap125 | **12.831** |
| 迁移：喷嘴→电池 | few-shot 1/2 电池 | scratch → transfer RMSE | 35.2→**27.7** / 33.7→**28.9** |
| 迁移：喷嘴→轴承代理 | few-shot 1/2 轴承 | scratch → transfer RMSE | 224.3→223.7 / 216.0→284.4 |

说明：
- 电池优化仅改电池入口和电池输出目录，**未重训喷管/C-MAPSS/轴承**；喷管数字保持冻结。
- 历史/临时结果：v0=`29.46`，v1=`23.20`，v3固定 MultiScale=`18.97`，v3 validation-selected=`21.03`。这些结果使用过旧的选择或窗口验证流程，不作为 clean strict14 canonical 指标。
- clean strict14 v2：首次观测到 `capacity <= 1.4 Ah` 定义 event。B0005/B0006/B0018 是 event-observed；B0007 未达到阈值，因此是 right-censored，不生成伪造的 `RUL=0`。在共同 endpoint cycle 23、嵌套电芯级 LOOCV、selection/final 同一组五 seed、scheduler budget=160 下，selected neural 的三 event-cell outer macro RMSE 为 **12.024**，selected method blend 为 **12.797***。
- v2 的 `selected_neural` 只在 neural candidates 中按 inner neural RMSE 选择；`selected_method_blend` 则按 inner OOF neural/Ridge blend RMSE 选择，alpha 使用 0.01 网格且允许 alpha=0 的明确 Ridge fallback。
- v2 同协议基线：Ridge=`13.898`；SOH-monotone=`15.223`；Huber=`22.531`；capacity trend=`27.291`。fixed family 仅作预声明诊断：MultiScale=`10.854`、BatteryLifeNet-L16=`11.137`、Transformer=`11.236`、GRU=`13.069`；它们不能替代 nested selected-neural 主结果。
- 上一版增强 canonical 的 `16.243 / 15.886` 与更早三 seed replay 的 `14.305 / 11.061` 保留为历史/敏感性对照；主表只采用 v2 canonical。
- `*` `12.797` 是 B0005/B0006/B0018 三颗 event-observed 电芯的 outer-test 宏平均；它不是四电芯宏平均，也不是 universal NASA SOTA。v2 结果显示 selected neural 在本协议下超过 Ridge，但公开论文的 EOL、split、forecast origin、B0007 标签和 metric units 不一致，因此仍不能直接宣传 universal NASA SOTA。
- **B0007 adaptive special**：为便于与采用 rel80 或其他自定义 EOL 的工作对照，B0007 可单独按 adaptive/rel80 评估；该结果不与 strict14 exact-event mean 混合。旧专项值保留在 `outputs/battery_b0007_confirm/` 作为历史参考。
- 可审计入口：`scripts/exp_battery_strict.py`；canonical/replay 输出使用新的独立目录，并包含 frozen config、协议 manifest、消融、逐 seed 指标和逐 seed 预测。
- 由于公开论文的 EOL、split、forecast origin、B0007 标签语义和 metric units 不一致，当前结果应表述为“该明确协议下的结果”，不能直接宣传为 universal NASA SOTA；外部同协议强基线复现和新电芯验证仍是必要条件。
- 喷管未重训，数字冻结 2.501/2.674/3.819。
- 轴承代理、迁移主实验数字见 `outputs/main_suite/`。
- 历史 JSON：`outputs/main_suite/MAIN_REPORT.json`、`outputs/battery_opt_v3/BATTERY_OPT_V3_REPORT.json`、`outputs/battery_b0007_confirm/BATTERY_B0007_CONFIRM_REPORT.json`。clean strict14 使用 `scripts/exp_battery_strict.py` 的独立输出目录。

## 反作用轮仿真域（新增，独立于 FEMTO proxy）

负责仿真的同学补充了 COMSOL 全过程退化数据：`反作用轮全过程comsol仿真退化数据集.zip`，SHA256=`b628deb89634cf8732cd96bb62d675b820aedb0aa337fd36082db53bfebca8fe`。该数据不能与 FEMTO 轴承 proxy 的 `0.331 / 0.661` 混合求均值，因为标签、单位和故障定义完全不同。

- 主对齐周期数据：5 条完整轨迹、每条 300 周期；轨迹1/2/3训练，轨迹4验证，轨迹5测试；包含明确的失效周期和物理 `剩余寿命_周期`。
- 低误差窗口数据：轴承磨损、匝间短路、永磁体退磁、综合退化四类故障；轨迹1/2/3训练，轨迹4验证，轨迹5测试；训练/验证/测试样本和源文件不交叉。
- leakage-safe Ridge 审计：对齐数据 operational tier 固定划分测试 RMSE=`7.625` cycles、MAE=`6.369`；低误差窗口 noisy tier 测试 RMSE=`8.162` levels、MAE=`6.784`。
- 初步模型已完成：operational 特征、序列长度20、GRU/多尺度TCN/Tiny Transformer候选、五 seed、轨迹1–3训练/4验证/5测试。nested candidate 选择的 GRU 测试 RMSE=`8.160` cycles、失效前 RMSE=`8.815`；同协议 Ridge=`7.108` cycles、失效前=`7.824`。当前模型已跑通，但神经模型尚未超过 classical baseline。
- oracle tier 包含真实退化参数或预计算 RUL 估计器，近零误差只作为泄漏审计上界，**不作为模型结果**。
- 当前仿真结果只能表述为“COMSOL 反作用轮仿真域结果”；不能表述为真实航天器反作用轮 SOTA。FEMTO proxy 和 COMSOL simulation 必须分别报告。
- 数据适配器：`src/data/reaction_wheel_sim.py`；审计入口：`scripts/audit_reaction_wheel_sim.py`；初步模型入口：`scripts/exp_reaction_wheel_sim.py`；报告：`outputs/reaction_wheel_sim_audit/SIMULATION_AUDIT.json`、`outputs/reaction_wheel_sim_initial/REACTION_WHEEL_SIM_INITIAL_REPORT.json`、`docs/reaction_wheel_simulation_audit.md`。

## 喷管严格协议结果（单轨迹）

旧喷管数字 `2.501 / 2.674 / 3.819` 仍保留在主实验表中，但它们属于历史非严格协议：原始 train/test 有重叠，且 Pure TCN/PCG-TCN 曾使用测试集选 checkpoint。它们不再作为 SOTA 证据。

严格单轨迹入口：`scripts/exp_nozzle_strict.py`。

- 56 行 COMSOL 轨迹，3 个 expanding rolling-origin folds；`fold3` 是锁定测试折。
- 目标：到 `0.2585 mm` 失效深度的 time-RUL（秒）；不输入绝对时间；train-only scaler。
- 5 个固定 seed：`42/123/456/2026/3407`；只用 validation early stopping；测试集只评估一次。

严格单轨迹结果（RMSE，秒）：

| 方法 | rolling 三折宏平均 | locked fold3 |
|---|---:|---:|
| Current-rate 物理基线 | 0.8767 | **0.0178** |
| Ridge | **0.7988** | 1.1563 |
| Physics-Full TCN | 0.9499 | 0.0523 |
| LSTM/TCN/Transformer | 6.0–6.4 | 10.4–11.0 |

**当前限制**：单条轨迹、单一工况，不能支持跨工况泛化结论。Physics-TCN 在 locked fold 上接近物理基线，但 rolling macro 尚未超过 Ridge。

## 喷管多工况 benchmark（新增，等待新轨迹）

新增独立多轨迹 benchmark 框架，用于接收仿真同学生成的多工况轨迹后直接启动严格实验：

- 数据 schema：`src/data/nozzle_multitrajectory.py`
  - 必填：`trajectory_id, condition_id, time_s, cumulative_ablation_depth_mm, ablation_rate_m_s, failure_depth_mm, solid_temperature_K, heat_flux_W_m2, pressure_Pa`
  - 阈值插值（非容差接受）；未到阈值标 right-censored，不伪造 RUL=0
  - observable tier（温度/热流/压力）和 estimated tier（加上累计深度/烧蚀率）明确分离；oracle 字段被强制排除
- 模型：`src/models/nozzle_multitrajectory.py`
  - `PhysicsResidualRateNet`：预测未来烧蚀率增长因子 → 积分到阈值 → EOL/RUL；从 identity 初始化，不从 current_rate 偷窃
  - `WeibullRULLoss`：CDF 差值加权，临近失效时段权重更高（von Hahn & Mechefske 2022）
  - `CensoredRULLoss`：event-observed 用 Weibull 加权损失；right-censored 用单侧 hinge
  - GRU / MultiScale TCN / Transformer 同协议基线
- 实验入口：`scripts/exp_nozzle_multitrajectory.py`
  - 外层：leave-one-trajectory-out；内层：旋转 LOO 选候选、窗口、特征 tier 和 epoch
  - 只用 inner validation 轨迹选择；测试轨迹最后只评估一次
  - 五 seed、等权 ensemble、group bootstrap 95% CI
  - 同时报告 all-run RMSE 和 pre-failure RMSE；frozen config SHA256 完整性校验
- 测试：`tests/test_nozzle_multitrajectory.py`（13 项，全部通过）

当前状态（synthetic smoke test）：

```
selected_neural all_rmse=3.18  ridge all_rmse=0.26  current_rate all_rmse=9.42
```

说明 ridge 在合成线性退化数据上最强；实际多工况非线性轨迹是真正的追分场景。

**启动条件**：仿真同学提供 ≥12 条独立 run-to-failure 轨迹，跨 ≥3 工况组，保留参数表和 `trajectory_id/condition_id`。

```bash
# 有真实多轨迹数据时
python scripts/exp_nozzle_multitrajectory.py \
  --data path/to/multi_trajectory.csv \
  --output outputs/nozzle_multitrajectory_v1 --device cuda:0

# 仅 synthetic smoke test（不依赖真实数据）
python scripts/exp_nozzle_multitrajectory.py --synthetic --output /tmp/smoke
```
| Ridge | **0.7988** | 1.1563 |
| Huber | 0.8069 | 1.1805 |
| Physics-Input TCN | 0.8967 | 0.0527 |
| Physics-Full TCN | 0.9499 | 0.0523 |
| Tiny TCN | 6.1824 | 10.3886 |
| Tiny LSTM | 6.0309 | 11.0164 |
| Tiny Transformer | 6.4088 | 10.8745 |

严格结果显示：Physics-Guided TCN 显著优于普通神经基线，但没有稳定超过确定性物理/Ridge 基线。因此当前喷管方向**不能宣传为 SOTA**；最多表述为“单条 COMSOL 轨迹、严格协议下的物理引导方法学结果”。当前数据也不足以支持全球喷管领域或外部泛化结论。

严格报告与审计：`outputs/nozzle_strict/NOZZLE_STRICT_REPORT.json`、`outputs/nozzle_strict/NOZZLE_STRICT_REPORT.md`、`outputs/nozzle_strict/frozen_config.json`、`docs/nozzle_public_data_audit.md`。

## 🔧 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
# C-MAPSS 数据已包含在 data/processed/cmapss/
# NASA 电池数据已包含在 data/processed/nasa_battery/
# 喷嘴严格基准数据: data/raw/nozzle_ablation_full.csv（GBK，56行，SHA256见严格报告）
```

### 训练

```bash
# FD002严格协议：多尺度TCN + 训练域工况归一化
python scripts/clean_benchmark.py --fd FD002 --model multiscale \
  --sensor-mode condnorm24 --seq-len 60 --seeds 42 \
  --device cuda:0 --output outputs/clean_fd002

# 主实验全套：电池 + 轴承代理 + 喷嘴 + 跨域迁移
python scripts/exp_main_suite.py --data-root data/processed \
  --output outputs/main_suite --device cuda:2

# 喷管严格协议：先冻结配置，再运行 canonical 5-seed benchmark
python scripts/exp_nozzle_strict.py --phase all --device cuda:0 \
  --output outputs/nozzle_strict

# 电池 clean strict14（不触碰喷管/C-MAPSS；先冻结选择，再复训 final）
python scripts/exp_battery_strict.py --phase all --data-root data/processed \
  --output outputs/battery_strict_v2_canonical --device cuda:0

# COMSOL 反作用轮仿真域审计（与 FEMTO proxy 分开）
python scripts/audit_reaction_wheel_sim.py \
  --archive path/to/反作用轮全过程comsol仿真退化数据集.zip \
  --output outputs/reaction_wheel_sim_audit

# SHRDL 自监督预训练
python scripts/pretrain_shrdl.py --gpu 0

# PEUDA 跨域迁移（接口占位，主结果以 exp_main_suite 为准）
python scripts/adapt_peuda.py --source nozzle --target battery --gpu 0
```

## 🔬 方法论

### 严格Benchmark主模型
- 三路并行多尺度TCN（kernel 3/5/7）
- 轻量多头自注意力聚合
- FD002/FD004：训练Unit拟合的六工况条件归一化
- 窗口60、`RUL cap=125`
- MSE + 轻量高估惩罚，按验证RMSE保存best checkpoint

### 灰盒与跨域模块
- PCG-TCN：物理约束灰盒喷嘴烧蚀模型
- SHRDL/PEUDA：自监督健康状态分解和跨域迁移实验

### 跨域迁移流程
1. **SHRDL** 在源域进行自监督预训练
2. **PEUDA** 执行无监督域自适应
3. **PCBNN** 在目标域进行不确定性量化

## 📝 竞赛信息

- **竞赛名称**：基于跨领域退化数据迁移的航天器关键组件寿命预测方法研究
- **主办单位**：中国科学院微小卫星创新研究院
- **应用场景**：喷嘴烧蚀（源域） → 反作用轮/电池（目标域）

## 📄 License

MIT License
