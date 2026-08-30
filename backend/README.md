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

> 注意：C-MAPSS论文对测试RUL是否同步应用RUL cap并不总是说明清楚，因此raw和cap125必须分栏报告，不能直接混比。

## 🔧 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
# C-MAPSS 数据已包含在 data/processed/cmapss/
# NASA 电池数据已包含在 data/processed/nasa_battery/
# 喷嘴烧蚀数据: data/raw/bimian_shaoshi.csv
```

### 训练

```bash
# FD002严格协议：多尺度TCN + 训练域工况归一化
python scripts/clean_benchmark.py --fd FD002 --model multiscale \
  --sensor-mode condnorm24 --seq-len 60 --seeds 42 \
  --device cuda:0 --output outputs/clean_fd002

# SHRDL 自监督预训练
python scripts/pretrain_shrdl.py --gpu 0

# PEUDA 跨域迁移
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
