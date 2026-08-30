# ASTRA 分支策略

## 主分支

`main` 是评委和复现人员应使用的主分支，固定承载：

- 主实验协议与结果：`scripts/exp_cross_transfer_v3.py`
- FEMTO -> COMSOL outer-fold 验证：`scripts/exp_femto_ims_to_comsol_outerloo_v5.py`
- 数据、Docker 环境、测试和审计材料
- 评委版技术报告及其生成脚本

当前主实验基线提交为 `86b405f`（2026-08-29 前已完成的主实验与复现材料）。软件界面、仪表板和演示脚本不得替换 `main` 的主实验协议或结果文件。

## 软件支线

软件部分作为工程演示层维护。建议使用 `feature/software` 或 `software-demo` 分支开发，目录和入口包括：

- `scripts/run_engineering_demo.py`
- `scripts/run_engineering_demo.ps1`
- `docs/engineering_demo.md`
- `outputs/engineering_demo/femto_ims_to_comsol_v5/`

软件支线合并前必须通过工程演示测试，并确认不修改主实验的切分、标签、scaler、outer-fold、指标和审计文件。

## 历史算法与仿真分支

`algorithm/*`、`simulation/*` 和旧的组件分支保留为历史开发线或专项实验线，不作为评委默认入口。它们的结果必须以对应协议和状态文件为准，不自动覆盖 `main`。

## 远端同步

当前本地 `main` 已与主实验提交一致。远端 GitHub 的默认分支设置属于仓库管理操作，需要在 GitHub 仓库设置中将默认分支切换为 `main`；远端同步前先确认新增软件分支名称，再执行：

```bash
git fetch origin --prune
git push origin main:main
git push origin feature/software:feature/software
```

不要删除 `algorithm/main` 或软件分支，除非团队已完成备份、评审和迁移确认。
