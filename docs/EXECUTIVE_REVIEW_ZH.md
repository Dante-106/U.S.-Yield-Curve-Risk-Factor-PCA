# 纽约固定收益交易与风控视角：PCA 代码深度审查与最终技术结论

## 一页结论

本工程已经从“单体研究 notebook”重构为可审计的 Python 3.10–3.13 工程，核心数学、时间顺序、金融单位、数据血缘、输出完整性和失败路径均被显式控制。就仓库声明的“冻结历史样本 PCA 研究与一阶 KRD 描述性诊断”范围，技术结论为 **SHARE WITH CAVEATS**；这不是机构模型验证、模型批准、投产许可或管理层风险接受。

当前唯一正确的业务动作是：**不使用该冻结输出做当前风险、限额、VaR/ES、FRTB、资本、交易、对冲或管理预测决策。** 数据截至 2025-12-26，相对 2026-08-26 纽约发布核验日期已陈旧 243 天；PCA 历史稳定性显著超阈值；没有预测 challenger 通过采用门槛；18 个独立区间校准指标全部 `INCONCLUSIVE`；示例 KRD 最近窗口 VaR/ES 有效尾部样本均低于 20，因此 KRD 风险结论为 **NEEDS REVISION**。

“全局最优”在模型风险中不能靠措辞证明。本次交付采用的可辩护最优准则是：在数据、用途与算力约束下，优先最小化错误采用、时间泄漏、单位错配、静默降级、结果篡改和治理误读，而不是最大化表面 PASS 数量。

## 原代码最严重问题

原 notebook 对展示用 components/scores 做了符号翻转，却没有同步修改 sklearn estimator 内部状态。因此图表、表格与后续 `transform`/`inverse_transform` 实际使用了不同因子定义。实测最大 transform 不一致为 211.99 bp，所谓 full-rank inverse 最大误差为 143.33 bp。这是 P0：任何基于因子暴露、情景、预测或风险映射的结论都可能引用错误符号坐标。

新实现使用单一对称特征分解对象统一维护中心、尺度、已定向 eigenvectors、scores、transform、inverse transform、物理 bp shock basis 与 one-sigma shocks；full-rank 重构最大误差约 `7.1e-14` bp。

原代码还存在以下系统性问题：边界周导致样本数漂移；缓存可能截断历史；配置与实际逻辑脱节；rolling 只看 explained variance；经济标签缺少歧义控制；forecast 不是严格 adoption protocol；区间可能带 post-selection/依赖误读；没有完整 KRD、residual、tail、stress、VaR backtest 与 hedge 控制；“自洽”被误称独立验证；缺乏 current/PIT/official-use 边界。

## 数据与 PCA 结果

冻结样本包含 6,783 个日度观测、11 个源期限；默认使用九个同步完整期限，形成 1,356 条周度曲线和 1,355 个周变动。第一条 level baseline 是 2000-01-07，变动区间是 2000-01-14 至 2025-12-26。

| 指标 | 结果 | 风控解释 |
|---|---:|---|
| PC1 / PC2 / PC3 explained variance | 77.6249% / 14.0276% / 5.2131% | Top-3 累计 96.8656% |
| Level / Slope / Curvature cosine | 0.9477 / 0.9488 / 0.9320 | 高于 0.70 similarity gate |
| dominance margin | 0.6791 / 0.6749 / 0.7781 | 高于 0.10；避免近似并列时强制命名 |
| Top-3 MAE / RMSE | 1.3421 / 1.9882 bp | 不能删除局部 KRD/residual risk |
| Top-3 p95 / 最大绝对误差 | 3.9039 / 22.7134 bp | 必须保留局部与历史压力情景 |
| expanding-OOS 最差期限 RMSE | 2.3783 bp | 低于 5 bp review level |

Level/Slope/Curvature 是模板识别约定，不是因果因子。若 similarity 或 dominance 失败，代码返回 `UNIDENTIFIED`，rolling/bootstrap 只用 `PC1/PC2/PC3 loading cosine`，不会通过列名继续传播经济标签。

## 稳定性与方法学结论

五年 rolling 对终端 full-sample basis 的最差 principal angle 为 71.173°，最小 aligned cosine 为 0.3935；相邻非重叠 sequential windows 的历史最差值为 73.903°/0.2833。最近窗口改善至 9.271°/0.9339，但最新 PASS 不能覆盖历史反证。因此 PC2/PC3 不可直接作为稳定的 limit taxonomy。

rolling 是 retrospective full-sample-reference diagnostic，不是历史时点告警。sequential windows 时间上隔离、没有使用监控窗口结束后的行，但仍来自同一 frozen/latest-revised vintage，不是 point-in-time replay。

13 周 circular block bootstrap 使用 2,000 次复制，每个 2.5% 尾部有 50 个期望顺序统计量；angle median/97.5% 为 3.0202°/7.9689°。4/13/26 周 sensitivity 的 97.5% angle 分别为 6.8342°/7.9460°/9.0982°。这些数值说明“给定全样本的抽样误差”较小，但不能推翻真实历史 regime instability。

2021-12-06 Treasury curve methodology 前后 top-3 最大 ordered subspace angle 为 7.8752°；PC1 one-sigma norm 从 28.3215 bp 上升到 36.2093 bp，比例 1.2785，超过 1.25 review level。由于市场 regime 与源方法同时变化，不能做因果归因。

current-EWMA 相对 structural 的三个 factor one-sigma ratio 为 1.0442/0.8152/0.8391，top-3 ordered angle 为 1.1167°/4.5377°/5.5977°；最大对称 sigma ratio 为 1.2267，低于 1.25 review level。该绝对 shock-scale 比 explained variance share 更能回答“当前风险水平是否改变”。

## 利率预测结论

所有模型严格 expanding-origin，且每一行满足 `training_end < target_date`。835 周 holdout 分为：

- selection：523 周，2009-12-31 至 2020-01-03；
- confirmation：52 周，2020-01-10 至 2020-12-31；
- independent interval evaluation：260 周，2021-01-08 至 2025-12-26。

selection curve RMSE：no-change 8.0363 bp、historical mean 8.0631、PCA AR 8.0558、PCA VAR 8.0533；所有 challenger 都更差。PCA VAR 在 confirmation 改善 2.2165%，但 one-sided Clark-West/HAC p-value 为 0.06793，且此前 selection 已失败，不能采用。最终的 `No-change benchmark` 只是“未发现预测 edge”的 fallback，不是批准模型。

独立 260 周边际区间中，80% band 最低覆盖率 68.4615%，95% band 最低 88.8462%，最差 gap 为 -11.5385pp；18/18 dependence-aware familywise block-bootstrap 状态均为 `INCONCLUSIVE`。80%/95% 边际 band 的全期限 joint-hit diagnostic 为 41.1538%/76.5385%，不能与边际 nominal coverage 直接比较，也不是 simultaneous curve band。

该五年窗口可用于发现严重失准，但没有足够功效认证正常覆盖。在乐观 IID 单侧 `alpha/18` 近似下，真覆盖等于 nominal 时通过概率约 4.9%/15.9%；正式双尾 `alpha/(2*18)` 口径更低，约 2.34%/9.40%。依赖会进一步降低有效样本。因此 `INCONCLUSIVE` 必须触发人工复核和禁止 production band，而不是机械改成 PASS 或 FAIL。

## 固定收益风险映射结论

DV01 定义为收益率下降 1 bp 时的正价格收益，因此一阶 P&L 为 `-DV01 @ shock_bp`。factor exposure 使用物理 bp-per-score basis；residual variance 直接对被省略的 full-rank factor exposure 求和，避免 `total-retained` 的消减误差，并与完整物理 covariance 做绝对/相对容差核验。

示例 KRD 不是实际头寸。其结果为：

| 指标 | 最近 520 周 | 全历史 1,355 周 |
|---|---:|---:|
| 99% finite-sample predictive VaR | $23.400m | $24.225m |
| rank / strict exception bound | 516 / 0.9597% | 1,343 / 0.9587% |
| 97.5% fractional empirical ES | $22.781m | $25.276m |
| VaR / ES tail mass | 5.2 / 13.0 | 13.55 / 33.875 |

只有全历史 ES tail mass 达到 20；最近窗口 VaR 和 ES 均未达到。rolling VaR 有 835 个回测点、8 个 exception、有限样本期望 8.013；Kupiec/independence/conditional coverage p-value 为 0.9962/0.6938/0.9254。8 个 exception 中 7 个集中于 2022–2025，aggregate PASS 只是低功效、regime-concentrated diagnostic。

全历史最差一阶损失为 2001-11-16 的 $47.625m。`±1/2/3σ` 是历史 covariance-scaled shock，不代表概率、return period 或 joint plausible scenario。代码没有 convexity/gamma、option、spread/basis、carry/roll、volatility、liquidity、funding、cross-curve 或 full revaluation。

对示例 KRD，structural full-covariance weekly modeled volatility 为 USD 9.069m，current-EWMA 为 USD 9.496m，上升 4.71%；对应 variance 约为 USD 82.2535tn 与 USD 90.1697tn。最终 notebook 和 CLI 均把这一绝对风险对比与 factor exposure 并列展示。

CLI 强制要求 curve、portfolio、position snapshot 与 sensitivity engine ID，但这些字段都只是调用者 self-attestation；程序仅校验必填性、语法和声明值，不能认证上游系统或证明 sensitivity lineage。KRD 若晚于 market as-of、早于超过七日、声明曲线/单位/bump/sign 不匹配、rolling VaR 未 PASS，或最近尾部样本不足，风险结论为 `NEEDS REVISION`。当前示例属于该状态，`--accept-warnings` 不能将其变成成功风险批准。

## 三个独立管理视角

### Fixed Income Risk Manager

保留完整 gross/net KRD、PCA factor 与 residual limits；structural 和 EWMA 风险必须并列看；历史最差与最新 stability 必须并列看；当前输出不进入 EOD risk pack，不触发 limit、trade 或 hedge。需要 actual positions、official sensitivities、official/hypothetical P&L、same-curve reconciliation、非线性 full revaluation 和 approved stress library。

### Model Validation Manager

数学实现、单位、符号、时间顺序和 fail-closed 路径在声明范围内可信。仍缺 PIT vintage、use-specific threshold calibration、机构独立性、正式 outcomes analysis、model inventory、owner/validator、issue governance、override expiry、change approval 与 retirement。alternate SVD 只是 implementation challenge。

### CEO / CRO

管理层页面必须首先显示 as-of/freshness、`NO CURRENT ACTION`、official P&L reconcile、limit headroom、full-revaluation stress、tail adequacy、exception concentration、owner、ticket、due date 和 approval authority。本工程当前不能回答“该不该加仓/对冲/调限额”，只能回答“在受控历史研究中，三因子压缩有效，但稳定性和预测证据不支持决策采用”。

## 工程与运行控制

工程包含严格不可变配置、跨字段样本/power 约束、bounded network/gzip/cache、POSIX/Windows crash-released cache lock、原子 cache/run commit、固定 artifact 名单、SHA-256 与 schema-v3 结构契约、严格 KRD/JSON schema、成功/失败 invocation ledger、确定性 notebook builder、source/data hash binding、Python 3.10–3.13 CI、sdist 解包测试和隔离 wheel/sdist smoke。

v3.1.0 已删除 package、scripts、tests 及 Notebook code cells 中全部 Python comment token；解释性意图改由清晰命名、显式常量、docstring、类型契约、测试与治理文档承载，并由 tokenizer regression gate 持续约束。对 v3.0.0 与 v3.1.0 分别完整运行默认 pipeline 后，覆盖 PCA、forecast、validation 与 structural/current KRD risk 的规范化序列化结果逐字节一致，SHA-256 均为 `8838bcbc07a60919726cffd668be589069630d07aa35e849799e95321e3e45d3`，未产生数学或金融结果漂移。

v3.2.0 修复了一个重大 Colab 验证缺口：此前所谓 Colab smoke 仍在完整仓库和已配置 import path 中执行，不能证明“只上传 notebook”可运行；在真正空目录中，首个 code cell 会因找不到 `src/` 与 `data/` 抛出 `FileNotFoundError`。新 notebook 内嵌确定性最小运行时，运行前核验 archive SHA-256、精确成员名单、压缩/解压大小、路径边界、implementation hash 与 snapshot hash，不进行 Git clone、网络下载或远程代码执行；完整仓库存在时仍优先绑定本地受控文件。另修复 Pandas 在缺少 Jinja2 时抛出 `AttributeError` 的兼容路径，表格自动降级为完整 DataFrame 而模型计算不中断。该路径已在无本项目安装、无 Jinja2 的全新环境中完成 17-cell 全链路执行。

v3.3.0 修复了截图所示的第二个 Colab 启动缺陷：v3.2.0 在第一单元格硬编码 Python `<3.13`，所以 Python 3.13 会在依赖、完整性和模型检查之前被人为终止。新版本把 package metadata、classifier 和 GitHub Actions 正式验证范围扩展至 Python 3.10–3.13；notebook 仅对低于 3.10 的运行时失败，对未来高于已验证范围的版本明确标注 `newer-than-validated Python runtime; guarded execution`，继续接受依赖、payload、source、data、计算与输出控制，不将其静默认证。使用真实 CPython 3.13.14、空目录、未安装本项目且未安装 Jinja2 的环境，17 个 code cells、59 个 outputs、5 个 PNG 全部完成且无错误；CPython 3.10.20、3.11.15、3.12.13 与 3.13.14 均各自通过 152 项测试和 Ruff。v3.2/Python 3.12 与 v3.3/Python 3.13 的完整规范化模型与 KRD 风险结果 3,718,192 字节逐字节一致，证明无数学或金融结果漂移。

为避免可变 manifest 自我证明，命中同一 content-addressed run 时仍完整重算，先独立导出 expected controls/artifact set，再逐文件核验 hash/schema 后复用目录。安全优先于跳过计算。

本地 ledger 不是签名、hash-chain 或 WORM；ticket label 不验证权限；依赖范围不是 lockfile/SBOM；retention、quota、free-space、stale-stage cleanup、备份恢复、DR、集中监控和 release attestation 仍由部署机构负责。

## 最终用途矩阵

| 用途 | 结论 |
|---|---|
| 冻结 H.15 历史 PCA 研究、教学、代码 peer review | 有条件可用，必须保留 caveats |
| “未发现 PCA forecast edge，保留 no-change” challenger 结论 | 可用 |
| 示例性一阶 KRD factor/residual/scenario 分解 | 仅诊断；当前 risk assessment NEEDS REVISION |
| 当前 desk risk、限额、EOD risk pack | 不可用 |
| 官方 VaR/ES、actual/hypothetical P&L backtest、PLA/FRTB、资本 | 不可用 |
| point forecast、interval band、交易信号 | 不可用 |
| hedge sizing 或自动交易 | 不可用 |
| pricing、SOFR/OIS、spread/basis/vol、MBS/options、nonlinear risk | 不可用 |

在扩大用途前，必须补齐 fresh/PIT approved data、正式头寸/估值/敏感度/P&L lineage、nonlinear benchmark、风险偏好校准阈值、机构独立验证、正式批准、集中不可变审计与完整运行治理。
