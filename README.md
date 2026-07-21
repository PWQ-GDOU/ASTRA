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
- **C-MAPSS 基准**：FD001=14.7, FD003=14.3, FD002≈28.8

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

## 📈 C-MAPSS Benchmark 结果

| 数据集 | 工况数 | 故障模式 | V2 TCN (RMSE) | Ensemble (RMSE) |
|--------|--------|----------|---------------|-----------------|
| FD001  | 1      | 1        | ~14.7         | —               |
| FD002  | 6      | 1        | ~29.5         | **28.8**         |
| FD003  | 1      | 2        | ~14.3         | —               |
| FD004  | 6      | 2        | —             | —               |

> ⚠️ FD002/FD004 存在~1.2% SNR（80:1工况噪声:退化信号比），构成数据驱动方法的物理极限。

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
# V2 TCN 训练 (FD002)
python scripts/train_v2.py --dataset FD002 --gpu 0

# SHRDL 自监督预训练
python scripts/pretrain_shrdl.py --gpu 0

# PEUDA 跨域迁移
python scripts/adapt_peuda.py --source nozzle --target battery --gpu 0
```

## 🔬 方法论

### V2 TCN 架构（主模型）
- 7层膨胀TCN块（dilation=2）
- 多头自注意力聚合
- 全局Z-score归一化
- Score-aware损失函数

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
