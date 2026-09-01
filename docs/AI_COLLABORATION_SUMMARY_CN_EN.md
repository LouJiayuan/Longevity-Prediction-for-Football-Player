# AI Collaboration Project Summary / AI 协作项目总结

> **历史版本 / Historical snapshot:** 本文件中的 27 August 分类结果已在伤病
> 数据溯源审计后被 1 September 2026 的无伤病变量重跑结果取代。请勿引用其中
> 的 all-XGBoost 数值；以 `METHODOLOGY.md` 和 `results/<task>/` 为准。 The
> classification values below are retained only as collaboration history and
> are not the corrected final evidence.

## 中文版

### 项目与最终研究设计

本项目研究职业足球运动员未来是否会出现持续性的出场时间下降。由于数据中没有权威的“真实能力衰退”标签，研究使用可复现的操作性结局：球员的赛季情境调整分钟数低于其严格前序正峰值的 65%，并连续两个相邻日历赛季保持低位。第一个低位赛季为 onset，第二个为 confirmation event。该结局描述出场机会的持续下降，不等同于能力、伤病或教练决策的单一原因。

最终冻结实验覆盖 2000-2025 年 13 个欧洲联赛，包括英、西、德、意、法的一、二级联赛，以及荷兰、葡萄牙和比利时的顶级联赛。样本包含 3,146,714 条球员比赛记录、113,890 场比赛、34,089 名球员、148,140 个球员赛季和 8,066 个首次确认事件。

单场分钟数使用两类 Elo 信息调整：双方平均 Elo 表示绝对比赛质量，对手与本方 Elo 差经标准 Elo 期望得分函数转换后表示相对对手难度。两项权重均为有界非线性函数，其乘积最终限制在 0.65-1.35。ClubElo 必须来自比赛日前至少一个日历日；同日记录单独审计但不用于特征或权重。Club rank 仅作为候选特征，不进入分钟数乘数。

研究报告两个不同的分类问题：

- `confirmation`：未来 2/5 年内是否确认持续下降；已经处于第一个低位赛季的球员仍可进入样本。
- `early_warning`：在下降尚未可见时，未来 2/5 年内是否开始一段随后被确认的低位序列；当前 candidate 被排除，负例还需额外一个赛季来确认 horizon 边界上的 onset。

### AI 工具、模型和选择原则

协作工具为 OpenAI Codex（GPT-5 系列 coding agent）。项目需要跨文件审查 Python 数据管线、时间切分、建模、诊断、图表和文档，因此它被用于工程协作与一致性检查，而不是作为科学事实来源。

分类比较 Logistic Regression、Random Forest、Gradient Boosting、XGBoost 和 LightGBM；生存分析比较 Cox PH、Weibull AFT、Random Survival Forest、Extra Survival Trees 和 Gradient Boosting Survival。Age-only 和 current-minutes-only 保留为朴素基线。所有缺失值插补均只从相应训练折学习；需要尺度处理的 Logistic、Cox 和 Weibull 模型使用训练集拟合的 z-score 标准化，树模型不标准化。

特征集和模型先在预设选择期确定，再进入锁定未来评估；查看锁定结果后不重新选择“赢家”。因此，锁定表中个别未选择算法分数略高时，也不应事后替换预先选择的模型。

### 研究者与 AI 的分工

研究者负责定义研究问题和研究边界；批准 65% 阈值、连续两年确认、13 联赛范围和 2000-2025 窗口；控制受限数据访问；评估方法取舍；解释结果；并对论文中的全部结论和引用负责。

Codex 协助审查数据与脚本依赖；实施严格前一日 Elo、训练折内插补、时间验证、双分类目标、基线和五模型比较；生成校准、联赛/级别稳健性、Permutation Importance、TreeSHAP 和代表性 held-out case 输出；运行一致性测试；以及整理可公开的 GitHub 代码、配置、聚合结果和说明文档。

### 关键挑战及处理

1. **没有外部 ground truth。** 研究明确使用 operational ground truth，并限制对其含义的解释。
2. **比赛情境不同。** 使用绝对质量和相对难度两个有界 Elo 权重；`alpha=0.25`、`beta=0.08` 是研究设定，不被表述为普遍常数。
3. **时间泄漏。** Elo 只能来自严格更早日期；未来目标、事后字段、峰值比例和标签构造字段不进入特征；插补和标准化只从训练数据估计。
4. **右删失。** 分类负例必须具有完整 horizon 观察；early-warning 负例还需额外的确认赛季。生存任务使用行政删失。
5. **确认与预警的概念差异。** confirmation 可利用已经出现的第一个低位赛季，因此比真正的 early warning 更容易；论文中必须分开陈述。
6. **可解释性边界。** 全局重要性、TreeSHAP、校准和 subgroup evaluation 描述预测关联及误差，不提供因果结论。
7. **数据公开限制。** GitHub 只包含代码、配置、聚合表和聚合图；原始/处理数据、模型、完整预测和球员级案例不公开。

### 最终结果

| 任务 | Horizon | Age-only AUC | 预先选择的 XGBoost AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|
| Confirmation | 2 年 | 0.699 | 0.811 | 0.579 | 0.138 |
| Confirmation | 5 年 | 0.680 | 0.736 | 0.849 | 0.189 |
| Early warning | 2 年 | 0.658 | 0.675 | 0.457 | 0.195 |
| Early warning | 5 年 | 0.663 | 0.693 | 0.845 | 0.186 |

Early-warning 相对年龄基线的 player-cluster bootstrap AUC 差异为：2 年 0.016（95% CI 0.001-0.032），5 年 0.029（95% CI 0.011-0.047）。这支持“存在有限但为正的增量预测信息”，不支持确定性个体判断。

生存分析中，Gradient Boosting Survival 的锁定 C-index 最高（0.748），Random Survival Forest 的 1-5 年 integrated Brier score 最低（0.216）。参数敏感性网格的 2 年 AUC 为 0.805-0.810，5 年为 0.713-0.727，最大事件行翻转率为 0.41%。敏感性结果用于检验稳定性，不用于在查看测试结果后挑选参数。

## English version

### Project and final design

This project studies whether professional footballers will experience a sustained future reduction in playing time. Because no authoritative label of “true ability decline” exists, the study uses a reproducible operational outcome: season-level context-adjusted minutes below 65% of the player's strictly prior positive peak in two adjacent calendar seasons. The first low season is the onset and the second is the confirmation event. This outcome describes sustained loss of playing opportunity; it is not a direct measure of ability, injury or a coaching decision.

The frozen experiment covers 13 European leagues from 2000 to 2025: the first and second tiers of England, Spain, Germany, Italy and France, plus the top tiers of the Netherlands, Portugal and Belgium. It contains 3,146,714 player-match rows, 113,890 matches, 34,089 players, 148,140 player-seasons and 8,066 first confirmed events.

Match minutes are adjusted using two Elo dimensions. Mean club Elo represents absolute match quality; the opponent-player club Elo gap is transformed through the standard Elo expected-score function to represent relative opponent difficulty. Both transformations are bounded and their product is clipped to 0.65-1.35. ClubElo must come from at least one calendar day before the match. Same-day records are audited but excluded. Club rank remains a candidate predictor and is not a minutes multiplier.

The study reports two distinct classification estimands:

- `confirmation`: whether sustained drop-off will be confirmed within 2/5 years; a player already in the first low season remains eligible.
- `early_warning`: whether a future first low season that is subsequently confirmed will begin within 2/5 years; current candidates are excluded and negatives require one extra confirmation season beyond the horizon.

### AI tool, models and selection principles

The collaboration tool was OpenAI Codex, a GPT-5-series coding agent. It was used because the work required cross-file review of Python data pipelines, temporal evaluation, modelling, diagnostics, figures and documentation. It served as an engineering collaborator and consistency checker, not as a source of scientific truth.

Classification compares Logistic Regression, Random Forest, Gradient Boosting, XGBoost and LightGBM. Survival analysis compares Cox PH, Weibull AFT, Random Survival Forest, Extra Survival Trees and Gradient Boosting Survival. Age-only and current-minutes-only models are retained as naive baselines. Missing-value estimates are learned only from the relevant training fold; Logistic, Cox and Weibull models use training-fitted z-score standardisation, while tree models are unscaled.

Feature sets and algorithms are chosen in a pre-specified selection period before locked future evaluation. A slightly higher locked score from an unselected alternative does not justify retrospectively changing the selected model.

### Division of work

The researcher defined the research question and scope; approved the 65% threshold, two-season confirmation rule, 13-league population and 2000-2025 window; controlled restricted-data access; judged methodological trade-offs; interpreted findings; and remains responsible for all dissertation claims and citations.

Codex helped audit data and script dependencies; implement strict prior-day Elo, fold-only preprocessing, temporal validation, two classification targets, baselines and five-model comparisons; generate calibration, league/tier robustness, permutation importance, TreeSHAP and representative held-out-case outputs; run consistency tests; and package public-safe code, configuration, aggregate results and documentation.

### Key challenges and controls

1. **No external ground truth.** The outcome is explicitly operational and its interpretation is limited accordingly.
2. **Unequal match context.** Two bounded Elo functions represent absolute quality and relative difficulty. `alpha=0.25` and `beta=0.08` are study choices, not universal constants.
3. **Temporal leakage.** Elo must be strictly earlier than the match; future, post-event, peak-ratio and label-construction fields are excluded; preprocessing is learned only from training data.
4. **Right censoring.** Classification negatives require complete horizon follow-up, with an extra season for early warning. Survival analysis uses administrative censoring.
5. **Confirmation versus warning.** Confirmation can use evidence from an observed first low season and is therefore easier than genuine early warning; the two claims are reported separately.
6. **Interpretability limits.** Importance, TreeSHAP, calibration and subgroup analyses describe predictive associations and errors, not causal mechanisms.
7. **Publication limits.** GitHub contains code, configuration and aggregate outputs only; raw/processed data, models, full predictions and player-level cases are excluded.

### Final results

| Task | Horizon | Age-only AUC | Pre-selected XGBoost AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|
| Confirmation | 2 years | 0.699 | 0.811 | 0.579 | 0.138 |
| Confirmation | 5 years | 0.680 | 0.736 | 0.849 | 0.189 |
| Early warning | 2 years | 0.658 | 0.675 | 0.457 | 0.195 |
| Early warning | 5 years | 0.663 | 0.693 | 0.845 | 0.186 |

For early warning, player-cluster bootstrap AUC differences versus age alone are 0.016 at two years (95% CI 0.001-0.032) and 0.029 at five years (95% CI 0.011-0.047). This supports limited but positive incremental information, not deterministic individual prediction.

Gradient Boosting Survival has the highest locked C-index (0.748), while Random Survival Forest has the lowest 1-5-year integrated Brier score (0.216). Across the weight-parameter grid, 2-year AUC ranges from 0.805 to 0.810, 5-year AUC from 0.713 to 0.727, and the maximum event-row flip rate is 0.41%. These are robustness checks, not a basis for post-test parameter selection.
