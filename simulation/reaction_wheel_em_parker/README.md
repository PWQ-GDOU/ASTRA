# 帕克反作用轮退化仿真 — 实体电磁模型 + 系统级 ODE 模型

## 概述

本仿真代码集来自反作用轮（Reaction Wheel）退化预测研究，包含两个层次的模型：

| 模型 | 类型 | 描述 |
|------|------|------|
| **二维实体电磁模型** | COMSOL 有限元 | stator-rotor-magnet 电磁场仿真，感应电流物理场 |
| **帕克对齐系统模型** | COMSOL 全局 ODE | 系统级电-机-热退化预测，基于 Parker et al. (2023) |

### 参考文献
Parker et al., "Advances in Space Research", 2023, Vol. 71, No. 6

### 软件版本
- COMSOL Multiphysics 6.4 (Build 293)

## 文件说明
- `README.md` — 项目总览
- `parameters.json` — 全部模型参数
- `em_model_spec.md` — 二维实体电磁模型详细规格
- `parker_system_model.md` — 帕克系统级 ODE 退化模型
- `governing_equations.md` — 控制方程（LaTeX 格式）
- `telemetry_schema.md` — 遥测数据格式说明
- `degradation_modes.md` — 退化模式与故障注入参数
