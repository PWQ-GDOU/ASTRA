# ASTRA 文书部门交接说明

更新时间：2026-08-26
项目仓库：`https://github.com/PWQ-GDOU/ASTRA.git`
工作区：`H:\项目代码目录\ZCodeProject`

## 交接内容

本仓库包含赛事规则、用户提供的数据附件、处理后的可复现数据、模型源代码、测试、
Docker 环境、完整实验报告、逐样本预测、工程寿命状态演示和数据哈希清单。总同步边界
见 `SYNC_MANIFEST.json`；输入数据说明见 `DATA_SOURCES.md`；赛事结果表见
`FINAL_RESULTS_TABLE.md`。

## 文书可引用的核心结果

### 喷嘴 -> NASA 电池

`outputs/cross_component_transfer_v3/` 的 strict14 结果采用 B0005、B0006、B0018
event-observed outer folds，B0007 作为 right-censored 诊断，不纳入精确 RMSE。
电池链路 N=1 transfer RMSE 为 51.119、相对 matched scratch 为 -2.04%；N=2
transfer RMSE 为 46.213、相对提升 1.94%。两种设置均未达到内部正向迁移证据标准，
文书应表述为诊断结果，不应写成迁移成功。

### FEMTO/IMS -> COMSOL 反作用轮仿真代理

`outputs/femto_ims_to_comsol_outerloo_v5_exploratory_fitonly_innerloo_full5x120_20260825/`
保存五条 COMSOL 仿真轨迹的 outer-fold 结果，固定 seeds 为 `42, 123, 456, 2026, 3407`。

| 设置 | Transfer RMSE | Matched scratch RMSE | 相对提升 | 胜出 outer folds | 适合表述 |
|---|---:|---:|---:|---:|---|
| N=1 | 7.575 | 8.047 | 5.87% | 3/5 | diagnostic，未超过 Ridge |
| N=2 | 4.974 | 5.366 | 7.30% | 4/5 | 内部 strict positive evidence |
| N=3 | 6.149 | 6.577 | 6.51% | 3/5 | 内部 strict positive evidence |
| N=1,2,3 聚合 | 6.233 | 6.664 | 6.46% | 10/15 | 内部 strict positive evidence |

这里的 COMSOL 是反作用轮**仿真退化代理**，不是真实反作用轮飞行遥测。该 v5 运行
标记为 `exploratory_post_hoc_replication`，并且要求新的、未参与条件设计的 outer
holdout 进行确认；因此只能写成“探索性严格协议证据”或“内部验收通过”，不能写成
已经完成官方盲测或工程放行。

## 可审计性和复现

- `PROTOCOL.json`：数据切分、scaler、target prior、adapter、early stopping 和选择规则。
- `DATA_MANIFEST.json`：输入文件、哈希和数据域说明。
- `PREFLIGHT.json`：泄漏检查、方向检查、输入形状和有限值检查。
- `ACCEPTANCE.json`：逐设置正向证据判定。
- `OUTER_FOLD_SUMMARY.csv`：逐 outer fold 的指标。
- `PREDICTIONS.csv`：最终预测及审计所需字段。
- `SUPERVISOR_STATUS.json`：长实验完成状态。
- `outputs/engineering_demo/femto_ims_to_comsol_v5/index.html`：离线工程寿命状态回放。

Docker 的 canonical full 和独立第二次 full 位于
`outputs/reproducibility/femto-ims-to-comsol-20260821T010000Z/` 与
`outputs/reproducibility/femto-ims-to-comsol-20260821T020000Z/`。两次运行的
`MACRO.csv`、`PREDICTIONS.csv`、`PROTOCOL.json` 和 `DATA_MANIFEST.json` 哈希一致；
Docker 报告记录了 Python/PyTorch/NumPy/scikit-learn 版本、镜像摘要、输入哈希及
`91 passed` 的只读数据挂载测试。

从仓库根目录运行：

```powershell
docker compose up --build engineering-demo
```

运行严格迁移复现：

```powershell
.\scripts\reproduce_femto_ims_to_comsol.ps1
```

## 数据哈希

下列哈希与用户提供附件核验一致：

| 材料 | 仓库路径 | SHA-256 |
|---|---|---|
| 赛事规则 PDF | `docs/competition/competition_rules.pdf` | `b610e21251805e5e345263cb98721ea3a5f2f0a80c00945e8795ee6e2987cf32` |
| COMSOL 反作用轮数据 | `data/raw/competition/reaction_wheel_comsol_degradation.zip` | `b628deb89634cf8732cd96bb62d675b820aedb0aa337fd36082db53bfebca8fe` |
| N-CMAPSS DS01 | `data/raw/competition/ncmapss_ds01.zip` | `e595250537f04bb4f374946dc85070d4635aba3d513ed32b3796a6448d239969` |
| 喷嘴数据 | `data/raw/competition/nozzle_ablation_quasi_steady_long.csv` | `2d1b4d906b8832e84f802eea449eb1ec0d744911604bd37cbfa70241e807a7c2` |
| Weibull 参考包 | `third_party/weibull-knowledge-informed-ml-master.zip` | `28d9d896ac45a605e75550ce6f703896c25183aff3a8e36c719a5c02db24e72e` |
| FEMTO 数据 | `data/processed/femto_bearing.zip` | `e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f` |
| N-CMAPSS 完整处理包 | `data/processed/ncmapss/N-CMAPSS_full.zip` | `121e547c56739cd7fe46468a2ec420297fd96c61a2813eb7af8e72f295060dab` |
| IMS 紧凑处理数据 | `data/processed/ims_processed/` | `887eea9cdb541ff40de8164f1913f9a37d22b12e3c48cc0eaa2d6dc7c100f4e6` |

## 写作边界

1. “当前评分 105/105”只能写为团队按赛事评分条目做出的证据支持自评上限，不能写成组委会正式评分。
2. 不要把 FEMTO 机械退化代理、COMSOL 仿真数据和真实反作用轮遥测混为同一数据源。
3. 不要把电池链路写成已证明正向迁移；其 strict v3 结果仍是 diagnostic。
4. 不要把 v5 探索性后验复现写成未见盲测；应保留 `requires_unseen_outer_holdout_confirmation` 的限定。
5. 论文或申报书中的精确数值应优先引用 `FINAL_RESULTS_TABLE.md` 和对应输出目录中的机器可读报告。