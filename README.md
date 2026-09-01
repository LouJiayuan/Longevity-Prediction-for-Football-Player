# Football Career Drop-off Prediction from Context-Adjusted Minutes

This repository contains the reproducible code path for an academic project on forecasting sustained reductions in professional football playing time. It uses only playing-time history, age, ClubElo context and a small set of audit/availability fields. It does not use goals, assists or other performance outcomes. The Manchester United-supplied source data and player-level prediction exports are not included.

The corrected, injury-free experiment regenerated on 1 September 2026 covers 13 European first- and second-tier leagues from 2000 to 2025. It contains 3,146,714 player-match rows, 113,890 matches, 34,089 players, 148,140 player-seasons and 8,066 first confirmed events, and reports two classification questions plus survival analysis. The legacy injury CSV was a synthetic test fixture and no injury-derived field enters any final label, feature set, model or result.

- **Confirmation:** will a sustained drop-off be confirmed within 2 or 5 years? A player already in the first low season remains eligible.
- **Early warning:** will a future first low season, subsequently confirmed by another adjacent low season, begin within 2 or 5 years? Current low-season candidates are excluded.
- **Survival:** what is the probability of remaining free from confirmed drop-off through each future horizon?

## Research design

For a player-club with Elo `E_c` facing an opponent with Elo `E_o`, match minutes are adjusted as follows:

```text
Q = (E_c + E_o) / 2
w_quality  = 1 + beta * tanh((Q - E_ref) / 200), beta = 0.08

g = E_o - E_c
p_opponent = 1 / (1 + 10^(-g / 400))
w_opponent = 1 + alpha * (2 * p_opponent - 1), alpha = 0.25

adjusted_match_minutes = raw_minutes * clip(w_quality * w_opponent, 0.65, 1.35)
```

ClubElo is joined using the latest observation from a **strictly earlier calendar day**. Same-day records are audited but excluded. `w_quality` captures the absolute level of both clubs and `w_opponent` captures relative opponent difficulty. A domestic cup match therefore inherits its weight from the actual clubs rather than a fixed cup label. Club rank is retained as a possible model feature but is not multiplied into minutes.

Direct prior-day Elo is available for 98.83% of player-club sides and 98.83% of opponent sides before fallback. Missing direct values follow the documented league-season-median then Big Five-reference hierarchy.

A first confirmed drop-off occurs when adjusted minutes are below **65% of the player's strictly prior positive peak in two adjacent calendar seasons**. The event is dated at the second low season; the first is the onset. The 50%, 60% and 70% definitions are sensitivity checks only.

## Locked experiment

All imputation is estimated inside each temporal training fold. Future, post-event and direct peak-ratio fields are excluded from model inputs.

Infinite values are converted to missing; training-fold medians are used for imputation and training-constant columns are removed. Logistic Regression, Cox PH and Weibull AFT are standardised with training-fitted z-scores; tree models are not scaled.

| Task | Selection period | Locked evaluation |
|---|---|---|
| Classification | target-end 2018-2021 | target-end 2022-2023 |
| Survival | origins 2013-2014; outcomes through 2019 | origins 2019-2020; outcomes through 2025 |

Classification compares Logistic Regression, Random Forest, Gradient Boosting, XGBoost and LightGBM. Survival compares Cox PH, Weibull AFT, Random Survival Forest, Extra Survival Trees and Gradient Boosting Survival.

## Headline results

| Classification task | Horizon | Selected model | Age-only AUC | Selected AUC | AP | Brier |
|---|---:|---|---:|---:|---:|---:|
| Confirmation | 2 years | Random Forest | 0.699 | 0.810 | 0.576 | 0.139 |
| Confirmation | 5 years | Gradient Boosting | 0.680 | 0.734 | 0.847 | 0.188 |
| Early warning | 2 years | Random Forest | 0.658 | 0.676 | 0.460 | 0.195 |
| Early warning | 5 years | XGBoost | 0.663 | 0.692 | 0.844 | 0.186 |

The confirmation task is easier partly because it includes players already in the first low season. The early-warning result is the stricter prospective claim and shows a smaller, but positive, improvement over age alone. For survival, Gradient Boosting Survival produced the strongest locked C-index (0.748) and 2-year dynamic AUC (0.784), while the pre-selected Random Survival Forest had the lowest integrated Brier score (0.216).

Model and feature-set choices were frozen from the selection period. Other algorithms or feature sets may score slightly higher in the locked tables; they were not retrospectively promoted after test inspection.

Across the 25 `alpha/beta` cells, the adjusted-trajectory AUC ranged from 0.805 to 0.810 at 2 years and 0.713 to 0.727 at 5 years; the maximum event-row flip rate was 0.41%. This is robustness evidence, not a basis for selecting parameters after observing test performance.

## Verified injury-history extension

The legacy `clinical` and `clinical_extended` feature sets are cancelled: they were built on a synthetic injury fixture. `src/38_verified_injury_feature_comparison.py` repeats the same temporal protocol with a ten-field block derived from an audited public Transfermarkt-ID injury history (CC0). These variables are availability-reporting proxies, not clinical records.

| Task | Horizon | Base AUC | With injury AUC | Difference |
|---|---:|---:|---:|---:|
| Confirmation | 2 years | 0.811 | 0.811 | -0.001 |
| Confirmation | 5 years | 0.738 | 0.741 | +0.003 |
| Early warning | 2 years | 0.672 | 0.671 | -0.001 |
| Early warning | 5 years | 0.692 | 0.697 | +0.006 |

Survival comparison shows an essentially unchanged C-index, roughly +0.002 five-year AUC and 0.001-0.0015 lower integrated Brier score. The primary full-period model stays injury-free because reporting intensity is heterogeneous over time: event-start prevalence among panel player-seasons is 0.5-1.1% in 2000-2006, 8.8-21.9% in 2007-2015 and 29.0-43.1% in 2016-2024. Injury is reported as a secondary long-horizon ablation, not a new primary model. See [docs/VERIFIED_INJURY_FEATURE_COMPARISON_2026-09-01.md](docs/VERIFIED_INJURY_FEATURE_COMPARISON_2026-09-01.md) and [docs/INJURY_SOURCE_RECONSTRUCTION_2026-09-01.md](docs/INJURY_SOURCE_RECONSTRUCTION_2026-09-01.md).

## Sequential practical application

The 2/5-year confirmation experiment above is a research benchmark. The operational layer in `src/37_sequential_player_application.py` routes each player by observed state so the two scores cannot be read as parallel clocks:

| Observed state at the landmark | Model applied | Question answered |
|---|---|---|
| No current low season | Early warning, cumulative horizons 1-5 | When in the next five years might a subsequently confirmed decline begin? |
| First low season below 65% | Conditional confirmation, next season only | Will the adjacent season also stay below 65%? |
| Decline already confirmed | No new forecast | The state is observed; review cause, role and recovery. |

Percentile bands rank review priority among players in the same observed state. They are not automated recruitment, contract or medical decisions. Aggregate metrics and the routing audit are in `results/practical_application/`; named-player score tables and fitted deployment models are restricted and excluded. See [docs/PRACTICAL_APPLICATION_FRAMEWORK_2026-08-29.md](docs/PRACTICAL_APPLICATION_FRAMEWORK_2026-08-29.md).

## Repository layout

```text
src/        Canonical pipeline and sensitivity scripts
config/     League scope and temporal-validation configuration
data/       Data contract only; raw and processed data are ignored
results/    Small aggregate tables from the final run, the verified injury
            extension and the sequential application layer
figures/    Aggregate visualisations for both classification tasks, the injury
            extension and sensitivity checks; no named cases
tests/      Core target, strict-Elo and injury-alignment tests
docs/       Method, injury-provenance, practical-application, data-access and
            AI collaboration notes
```

## Reproducing the pipeline

Use Python 3.12 where possible. `scikit-survival` may be easiest to install through conda/mamba on some platforms.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python src/01_build_weighted_minutes.py
python src/02_build_dropoff_labels.py
python src/03_feature_engineering.py
python src/08_build_model_targets.py
python src/29_final_experiment.py --task both --force-selection
python src/30_final_interpretability.py --task confirmation
python src/30_final_interpretability.py --task early_warning
python src/31_downstream_interpretability_figures.py --task confirmation
python src/31_downstream_interpretability_figures.py --task early_warning
python src/37_sequential_player_application.py
pytest -q
```

The verified injury extension runs separately from the primary path:

```bash
python src/00_align_longitudinal_injuries.py --download
python src/38_verified_injury_feature_comparison.py --task both
python src/39_summarize_verified_injury_results.py
```

Sensitivity scripts `33` to `36` are separate from the locked primary experiment. Run script `35` before script `36`, because the matched raw-feature comparison consumes the scenario targets produced by `35`. The reported LightGBM run retains the code's default `subsample_freq=0`, so its stored `subsample=0.85` does not activate row bagging; this is documented without changing the locked result.

Before running, read [data/README.md](data/README.md). The scripts require an authorised local copy of the raw inputs and write generated data to `data/processed/`, which is excluded from version control.

## AI collaboration disclosure

The researcher defined the question, controlled data access, made methodological decisions and reviewed outputs. OpenAI Codex helped inspect and refactor the pipeline, implement leakage controls and validation, run diagnostics, and prepare documentation. It was not used as an authority for scientific truth; final methodology and claims remain the researcher's responsibility. A bilingual account is available in [docs/AI_COLLABORATION_SUMMARY_CN_EN.md](docs/AI_COLLABORATION_SUMMARY_CN_EN.md).

## Data and ethical note

The underlying data were supplied for dissertation research and are not redistributable. This repository includes code, configuration, aggregate results and figures only. Any reuse requires independently obtained, appropriately licensed data and a review of applicable privacy, contract and competition-data restrictions.
