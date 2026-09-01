# Methodology

## 1. Data audit and analytical population

The canonical analysis includes the first and second tiers in England, Spain, Germany, Italy and France, plus the top tiers of the Netherlands, Portugal and Belgium. The modelling window is 2000-2025 because the supplied appearance database extends much further back, but usable ClubElo coverage begins in 2000. Third-tier data remain outside the primary experiment because pre-match ClubElo coverage is insufficient for the same weighting definition.

The rebuilt dataset contains 3,146,714 player-match rows, 113,890 matches, 34,089 players and 148,140 player-seasons. Before fallback, direct prior-day Elo coverage is 98.83% for player clubs and 98.83% for opponents. All joins, season mappings, league scope and missing-source indicators are retained for audit.

## 2. Strict pre-match Elo

For every club-side in match `j`, ClubElo is the latest observation whose calendar date is strictly earlier than the match date:

```text
E_club,j = Elo(club, max{d : d < match_date_j})
```

The implementation uses a left insertion point on normalized calendar dates. It asserts `match_date - elo_date >= 1 day` for every direct match and separately records whether a same-day value existed. In the current data, a same-day record existed for 228,748 club-sides (96.67%), so the explicit date rule is methodologically material even though downstream labels changed only slightly.

Missing direct ratings follow an explicit hierarchy: the relevant league-season median where available, then the season-specific Big Five reference. In the retained final rows the realised Elo sources are direct and league-season median; the final reference fallback remains a guarded code path. Unknown clubs are not assigned a universal rating below every covered league because missingness does not itself prove low strength.

## 3. Context-adjusted exposure

For match `j`, player `p` and season `s`, let `m_pj` be observed minutes, `E_cj` the player-club Elo, `E_oj` opponent Elo and `E_ref` the seasonal median Elo of the Big Five reference leagues:

```text
Q_j = (E_cj + E_oj) / 2
w_quality,j = 1 + 0.08 * tanh((Q_j - E_ref) / 200)

g_j = E_oj - E_cj
p_j = 1 / (1 + 10^(-g_j / 400))
w_opponent,j = 1 + 0.25 * (2p_j - 1)

AM_pj = m_pj * clip(w_quality,j * w_opponent,j, 0.65, 1.35)
AM_ps = sum_j AM_pj
```

The expected-score expression is the standard Elo logistic transformation. Its equivalent form is `2p_j - 1 = tanh((ln 10 / 800) * g_j)`. The two bounded value functions represent different dimensions: absolute match quality and relative opponent difficulty. Their product treats them as joint exposure components and returns exactly one when both are neutral. `clip(x, a, b)` truncates only values outside `[a,b]`; it prevents one match from receiving an extreme multiplier.

`alpha=0.25` and `beta=0.08` are pre-specified study choices, not universal football constants. A 5 x 5 sensitivity grid evaluates `alpha=0.15-0.35` and `beta=0.04-0.12`. Club rank is available as a predictor but is not a third minutes multiplier, because absolute match quality and opponent gap already encode the relevant club-strength information.

## 4. Operational drop-off outcome

For season `s`, the reference peak is the largest positive adjusted-minutes value strictly before `s`. Eligibility requires at least two observed historical seasons and a positive prior peak. A low season satisfies:

```text
AM_ps / max(AM_p,t for t < s) < 0.65
```

Drop-off is confirmed only when two adjacent calendar seasons are low. The onset is the first low season and the confirmation event is the second. Each player contributes only the first confirmed event. Raw minutes remain an audit field but do not define the primary label. Thresholds of 50%, 60% and 70% are sensitivity analyses rather than additional outcomes selected by AUC.

The rebuilt primary label contains 8,066 first confirmed events. Event counts increase monotonically from 5,839 at 50% to 8,714 at 70%, as expected.

## 5. Two classification estimands

The code now reports two intentionally different prospective questions:

| Task | Landmark eligibility | Positive target | Negative follow-up requirement |
|---|---|---|---|
| Confirmation | all pre-confirmation player-seasons | future `dropoff_event` within 2/5 years | observed continuously through horizon |
| Early warning | additionally requires current `dropoff_candidate=0` | future confirmed `dropoff_onset` within 2/5 years | observed through horizon plus one confirmation season |

The extra season for an early-warning negative prevents a low season at the horizon boundary from being mislabeled as a non-event before its possible confirmation is observed. The confirmation dataset contains 85,135 landmarks; the early-warning dataset contains 55,829.

## 6. Preprocessing, features and leakage control

Prospective feature ladders compare age only, minutes only, age/career, age plus minute trajectories, age/minutes/Elo and the full feature set. Features include age, current and lagged adjusted minutes, career exposure, games played, Elo context, broad position groups, competition shares, interactions and data-quality indicators.

An injury-event CSV used in early development was confirmed to be a synthetic test fixture and was removed. Reconstructed public injury sources were also withheld because their event/aggregate units, coverage and redistribution conditions were not sufficiently homogeneous for the locked multi-league analysis. No injury-derived variable enters the outcome, predictors, feature selection, model fitting or reported results; stale `injury_*` columns are defensively blocked by the shared feature selector.

Future targets, post-event fields, direct peak ratios and label-construction columns are excluded. Infinite numeric values are first changed to missing. Numeric medians are learned only from each temporal training fold; a feature whose training median is itself missing receives zero, and columns constant in the training fold are removed before fitting. The same learned values and retained columns are then applied to validation and locked-test rows.

Logistic Regression, Cox PH and Weibull AFT use z-score standardisation learned from training data. Tree-based models use the imputed values without scaling. Position is represented by one-hot indicators and age-position interactions; no target encoding is used. Selection caches are task-specific and validated by a SHA256 signature of the dataset, configuration, horizons, feature sets and model list.

## 7. Model comparison and temporal evaluation

Classification compares Logistic Regression, Random Forest, Gradient Boosting, XGBoost and LightGBM. Feature/model selection uses outcomes ending in 2018-2021; locked evaluation uses outcomes ending in 2022-2023. AUC evaluates ranking, average precision reflects precision-recall performance under class imbalance, and Brier score evaluates probability error. Calibration, league subgroup results, tier robustness and permutation importance are produced separately for each task; TreeSHAP is produced only where the pre-selected final classifier is XGBoost.

Hyperparameters are fixed in `src/29_final_experiment.py`, and randomised estimators use seed 42. One implementation detail matters for exact reproduction: the stored LightGBM specification sets `subsample=0.85` but leaves `subsample_freq` at its library default of zero, so LightGBM row bagging is inactive in the reported run; column subsampling remains active. This is documented rather than altered after locked evaluation.

Survival landmark analysis compares Cox PH, Weibull AFT, Random Survival Forest, Extra Survival Trees and Gradient Boosting Survival. Selection origins are 2013-2014 with outcomes through 2019; locked evaluation origins are 2019-2020 with outcomes through 2025. C-index and dynamic AUC evaluate ranking, while integrated Brier score evaluates survival-probability error.

## 8. Main interpretation

The corrected injury-free run selected `age_minutes_elo` for confirmation and `full_model` for early warning. Confirmation selected Random Forest at two years (AUC 0.810) and Gradient Boosting at five years (0.734), compared with age-only AUCs of 0.699 and 0.680. Early warning selected Random Forest at two years (0.676) and XGBoost at five years (0.692), versus age-only 0.658 and 0.663. The difference between tasks is substantively important: confirmation can use evidence from an already-observed first low season, whereas early warning asks the model to act before that signal appears.

Sensitivity results are stable across the weight grid: 2-year AUC 0.805-0.810, 5-year AUC 0.713-0.727 and at most 0.41% event-row flips. Replacing raw with adjusted trajectories alone gives only a small and inconsistent AUC change; the stronger claim supported by the ablation is that the complete Elo-contextualised feature system adds information, especially at five years. No additional uncertainty module is introduced in this revision.
