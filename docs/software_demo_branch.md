# 软件演示分支说明

评委默认从 `main` 查看主实验、数据协议、Docker 复现和审计结果；软件工程演示位于同仓库的 `software-demo` 分支。

软件分支的入口：

- `scripts/run_engineering_demo.py`
- `scripts/run_engineering_demo.ps1`
- `scripts/build_engineering_demo.py`
- `docs/engineering_demo.md`
- `outputs/engineering_demo/femto_ims_to_comsol_v5/`

该分支只负责把已锁定的 FEMTO/IMS -> COMSOL 代理实验结果转换为离线寿命状态回放，展示预测 RUL、健康状态和审计回放；不重新训练模型，不改变数据切分、标签、scaler、outer-fold、指标或主实验结论。COMSOL 仍是反作用轮机械退化仿真代理，不是真实在轨遥测。

仓库入口：`https://github.com/PWQ-GDOU/ASTRA`

建议复现顺序：先在 `main` 完成预检和主实验复现，再切换到 `software-demo` 构建演示页面。若 GitHub 尚未出现该分支，应先完成远端同步，再向评委提供分支名称。
