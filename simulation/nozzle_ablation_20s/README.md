# 喷管喉衬烧蚀仿真模型 — 反应流-固体传热-ALE 耦合

## 概述

本模型基于 COMSOL Multiphysics 6.4，对**固液火箭发动机喷管喉衬**在 20 秒热试车过程中的烧蚀行为进行多物理场耦合仿真。

### 物理场耦合
| 物理场 | 接口 | 描述 |
|--------|------|------|
| 湍流反应流 | ReactingFlowSST | SST k-ω 湍流 + 多组分输运 |
| 固体传热 | Solid Heat Transfer | C/C 复合材料喉衬导热 |
| 移动网格(ALE) | Moving Mesh | 烧蚀导致的内壁面后退 |
| 层流 | LaminarFlow | 固相区域无对流 |

### 几何结构（轴对称 2D）
```
前室(L=35mm,D=80mm) → 燃料通道(L=375mm,D=25mm) → 后室(L=50mm,D=80mm) → 喉部(D=15.875mm) → 喷管出口(D=26mm)
```

### 关键参数
详见 [parameters.json](./parameters.json)

### 软件版本
- COMSOL Multiphysics 6.4 (Build 293)
- 模型创建: 2026-07-16
- 最后修改: 2026-07-24

## 文件说明
- `parameters.json` — 全部模型参数
- `physics_setup.md` — 物理场详细设置
- `boundary_conditions.md` — 边界条件配置
- `solver_config.md` — 求解器配置
- `model_summary.md` — 模型技术摘要
