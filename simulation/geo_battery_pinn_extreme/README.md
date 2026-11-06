# GEO卫星电池PINN极端段退化仿真模型

## 概述

基于 COMSOL Multiphysics 6.4 的 GEO 卫星锂离子电池退化预测模型。模拟 18650 圆柱电池在 GEO 轨道阴影期极端工况下的 **SOH-内阻-温度退化**过程，输出 PINN（物理信息神经网络）训练所需的极端段特征。

### 模型架构

| 层次 | 物理场接口 | 维度 |
|------|-----------|------|
| **系统级 ODE** | 全局常微分方程 | 0D (集总参数) |
| **三维热场** | 固体传热 + 真空辐射 | 3D 实体 |

### 参考文献
- 卫星电池 SOH 退化半经验模型
- PINN 极端段特征提取框架
- GEO 轨道热环境 (阴影/光照周期)

### 软件版本
- COMSOL Multiphysics 6.4 (Build 293)
- 构建: 2026-07-26

## 文件说明
- `README.md` — 项目总览
- `parameters.json` — 全部参数与派生变量
- `governing_equations.md` — 控制方程 (LaTeX)
- `thermal_model.md` — 三维热模型与边界条件
- `degradation_model.md` — SOH-内阻退化机制
- `pinn_features.md` — PINN 极端段特征工程
- `telemetry_schema.md` — 输出遥测数据格式
