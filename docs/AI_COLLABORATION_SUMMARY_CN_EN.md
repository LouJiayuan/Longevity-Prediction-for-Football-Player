# AI Collaboration Project Summary / AI 协作项目总结

## 中文版

### 项目背景

本项目研究职业足球运动员的职业生涯持续性。由于数据中没有权威的“真实能力衰退”标签，研究不直接声称预测能力下降，而是以球员未来是否出现持续性的 `adjusted minutes` 下滑作为操作性预测目标。最终使用 2000-2025 年 13 个欧洲一级和二级联赛的数据。

单场原始分钟数先通过两类 Elo 信息调整：比赛双方平均 Elo 表示绝对比赛质量，对手与本方的 Elo 差表示相对难度。赛季 adjusted minutes 低于球员严格前序峰值的 65%，且连续两个相邻赛季维持低值时，第二个低值赛季被定义为首次确认的 drop-off。

### AI 工具、模型选择与原因

AI 协作工具为 **OpenAI Codex（GPT-5-based coding agent）**。选择它是因为任务涉及数据审查、跨文件 Python 管线、时间验证、多个模型、结果图表和汇报材料，而不是单一代码问题。Codex 的作用是协助实现和审查，不是替代研究判断或作为科学证据来源。

研究中的模型选择分为两类：

- 分类任务比较 Logistic Regression、Random Forest、Gradient Boosting、XGBoost 和 LightGBM，用于预测未来 2 年或 5 年内是否发生 drop-off。
- 生存分析比较 Cox PH、Weibull AFT、Random Survival Forest、Extra Survival Trees 和 Gradient Boosting Survival，用于预测首次 drop-off 的时间。
- Age-only 与 current-adjusted-minutes-only 被保留为朴素 baseline，用于检验复杂模型是否有额外价值。

最终分类模型选择 XGBoost：它在预设时间选择期表现最好，并在锁定未来测试中获得 2 年 AUC 0.812 和 5 年 AUC 0.738。生存模型保留为横向比较；在锁定测试被查看后不再重新选择“赢家”。

### 我与 AI 的分工

我负责定义研究问题、使用 adjusted-minutes 操作性标签、决定 65% 阈值与连续两年确认规则、决定 13 联赛和 2000-2025 的范围、控制曼联提供数据的访问、解释结果并对最终研究结论负责。

AI 负责审查数据与脚本依赖；将旧的 raw-minutes 门槛标签改为 adjusted-minutes-only；实现训练折内缺失值填补、时间切分、baseline、五模型比较、校准、联赛分组评估、Permutation Importance 和 TreeSHAP 案例分析；检查图表与 PPT；并将最终代码整理为 GitHub 目录。

### 核心挑战与解决方案

1. **缺少 ground truth。** 解决方式是明确使用 operational ground truth：它衡量持续性出场机会下降，而不把结果解释为能力、伤病或教练决策中的单一原因。
2. **不同比赛的分钟数不能直接相加。** 解决方式是使用有界、非线性的 Elo 权重：`tanh` 表示绝对比赛质量，标准 Elo 胜率形式表示相对对手难度，组合权重限制在 0.65-1.35；alpha 和 beta 通过敏感性分析检查。
3. **三级联赛 ClubElo 覆盖不足。** 虽然可以收集出场数据，但缺少赛前 Elo 会破坏对手强度设计。因此主实验只保留 13 个已对齐、可审计的联赛。
4. **时间泄漏与右删失。** 所有中位数只从每个时间训练折估计；负例必须完整观察对应 horizon；窗口末端和数据中断按删失处理，不当作“没有下滑”。
5. **复杂模型是否有实际增益。** 正式报告 age-only 和 minutes-only baseline。最终 XGBoost 相比年龄基线在 2 年 AUC 上提高 0.113（95% CI 0.098-0.126），在 5 年上提高 0.058（95% CI 0.045-0.073）。
6. **非技术解释。** 使用全局 feature importance、TreeSHAP 个案、校准曲线和一级/二级联赛 subgroup evaluation。它们解释预测关联，不能被表述为因果关系。

### 项目结果与 GitHub 说明

最终样本包括 148,140 个球员赛季和 8,064 个首次确认的 65% drop-off 事件。2 年预测中，Age-only AUC 为 0.700、Minutes-only 为 0.578、最终 XGBoost 为 0.812；5 年分别为 0.680、0.558 和 0.738。一级和二级联赛的 2 年 AUC 分别为 0.813 和 0.811。

本 GitHub 包只包含最终代码、配置、聚合指标、图和数据契约说明；不包含曼联提供的原始数据库、处理后的 player-level parquet、模型序列化文件、完整预测概率或任何球员级案例输出。带姓名的 TreeSHAP 案例图和案例表已从该副本移除。

## English version

### Project background

This project studies career longevity in professional football. Because the source data do not provide an authoritative label for a player's true loss of ability, the project does not claim to predict ability decline directly. Instead, it forecasts a sustained future reduction in `adjusted minutes` as an operational outcome. The final analysis covers 13 European first- and second-tier leagues from 2000 to 2025.

Raw match minutes are adjusted with two Elo signals: the average Elo of both clubs represents absolute match quality, while the opponent's Elo advantage represents relative difficulty. A player's first confirmed drop-off is the second of two adjacent seasons below 65% of their strictly prior positive adjusted-minutes peak.

### AI tool, model selection and rationale

The collaboration tool was **OpenAI Codex (a GPT-5-based coding agent)**. It was appropriate because the work was a repository-scale research pipeline involving data auditing, cross-file Python dependencies, temporal validation, multiple models, result figures and presentation material. Codex supported implementation and review; it was not an authority for scientific claims.

The research compares:

- Logistic Regression, Random Forest, Gradient Boosting, XGBoost and LightGBM for 2- and 5-year classification risk.
- Cox PH, Weibull AFT, Random Survival Forest, Extra Survival Trees and Gradient Boosting Survival for time to first drop-off.
- Age-only and current-adjusted-minutes-only naive baselines, so any value from a richer model is explicit.

XGBoost was selected for classification because it performed best in the pre-specified temporal selection period and achieved locked AUCs of 0.812 at two years and 0.738 at five years. Survival models remain a horizontal comparison; no winner was re-selected after viewing the locked test.

### Division of work

The researcher defined the question, chose the adjusted-minutes operational outcome, set the 65% threshold and two-season confirmation rule, fixed the 13-league 2000-2025 scope, controlled access to the Manchester United-supplied data, interpreted findings and remains responsible for all scientific claims.

Codex audited data and scripts; converted older raw-minutes-gated labels to adjusted-minutes-only; implemented fold-specific imputation, temporal splits, baselines, five-model comparisons, calibration, league subgroup evaluation, permutation importance and TreeSHAP player cases; checked figures and slides; and prepared the GitHub package.

### Core challenges and solutions

1. **No direct ground truth:** the target is stated as an operational ground truth for sustained loss of adjusted playing opportunity, not a single cause such as ability, injury or coaching choice.
2. **Non-comparable minutes:** bounded nonlinear Elo weights model absolute match quality and relative opponent difficulty. `tanh` is used for absolute quality, standard Elo expected score for relative difficulty, and the combined multiplier is clipped to 0.65-1.35. Alpha and beta are tested by sensitivity analysis.
3. **Insufficient third-tier ClubElo coverage:** the final experiment keeps only the 13 auditable, aligned leagues rather than assigning league-average Elo to most lower-tier clubs.
4. **Temporal leakage and right censoring:** medians are fit within each temporal training fold; negative labels require a fully observed horizon; data gaps and window endings are censored, not treated as no-event outcomes.
5. **Need to show incremental value:** final XGBoost improves AUC over age by 0.113 at two years (95% CI 0.098-0.126) and 0.058 at five years (95% CI 0.045-0.073).
6. **Interpretability:** permutation importance, TreeSHAP cases, calibration and subgroup evaluation make the outputs intelligible while remaining predictive rather than causal explanations.

### Outcomes and GitHub note

The final panel contains 148,140 player-seasons and 8,064 first confirmed 65% events. At two years, age-only, minutes-only and final XGBoost AUCs are 0.700, 0.578 and 0.812; at five years they are 0.680, 0.558 and 0.738. First- and second-tier two-year AUCs are 0.813 and 0.811.

The GitHub package contains only final code, configuration, aggregate tables, figures and the data contract. It excludes restricted raw data, processed player-level parquet, serialised models, complete prediction exports and player-level case outputs. The named TreeSHAP case figures and case table have already been removed from this copy.
