# Football Career Drop-off Prediction from Context-Adjusted Minutes

This repository contains the final code path for an academic project on forecasting sustained reductions in professional football playing time. It predicts an operational **drop-off** outcome from information available at the end of each player-season. The repository does not include the underlying Manchester United-supplied data or player-level prediction exports.

The final experiment covers 13 European first- and second-tier leagues between 2000 and 2025. It compares two prediction formulations:

- **Classification:** probability of confirmed drop-off within 2 or 5 years.
- **Survival analysis:** probability that a player has not dropped off by 2 or 5 years.

## Research design

Raw match minutes are converted into context-adjusted minutes. For a player club with Elo `E_c` facing an opponent with Elo `E_o`:

```text
Q = (E_c + E_o) / 2
w_quality  = 1 + beta * tanh((Q - E_ref) / 200), beta = 0.08

g = E_o - E_c
p_opponent = 1 / (1 + 10^(-g / 400))
w_opponent = 1 + alpha * (2 * p_opponent - 1), alpha = 0.25

adjusted_match_minutes = raw_minutes * clip(w_quality * w_opponent, 0.65, 1.35)
```

`w_quality` captures the absolute level of the two clubs; `w_opponent` captures relative opponent difficulty. Thus, a domestic cup match inherits its weight from the two actual clubs rather than a fixed cup label. Club rank is retained as a predictive feature but is not multiplied into adjusted minutes.

A player has a first confirmed drop-off when adjusted minutes fall below **65% of their strictly prior positive peak for two adjacent calendar seasons**. The event date is the second low season. Thresholds of 50%, 60% and 70% are reported as sensitivity checks only.

## Locked experiment

All numeric imputation is estimated inside each temporal training fold. Future, post-event and direct peak-ratio fields are excluded from model inputs.

| Task | Selection period | Locked evaluation |
|---|---|---|
| Classification | target-end 2018-2021 | target-end 2022-2023 |
| Survival | origins 2013-2014; outcomes through 2019 | origins 2019-2020; outcomes through 2025 |

Classification compares Logistic Regression, Random Forest, Gradient Boosting, XGBoost and LightGBM. Survival compares Cox PH, Weibull AFT, Random Survival Forest, Extra Survival Trees and Gradient Boosting Survival.

## Headline results

The final classification feature set was `clinical_extended` (32 prospective features) and the selected algorithm was XGBoost.

| Horizon | Age-only AUC | Minutes-only AUC | XGBoost AUC | Final minus age, 95% CI |
|---|---:|---:|---:|---:|
| 2 years | 0.700 | 0.578 | 0.812 | +0.113 [0.098, 0.126] |
| 5 years | 0.680 | 0.558 | 0.738 | +0.058 [0.045, 0.073] |

For survival, Gradient Boosting Survival had the strongest locked C-index (0.748) and 2-year dynamic AUC (0.784). The pre-selected Random Survival Forest had the best integrated Brier score (0.216) but weak locked ranking. Rich survival models did not clearly outperform age alone at five years; the formal conclusion is temporal instability rather than a post-hoc change of winner.

## Repository layout

```text
src/        Canonical pipeline scripts
config/     League scope and temporal-validation configuration
data/       Data contract only; raw and processed data are ignored
results/    Small aggregate tables from the final run
figures/    Final aggregate visualisations
docs/       Method, data-access and AI collaboration notes
```

## Reproducing the pipeline

Use Python 3.12 where possible. `scikit-survival` may be easiest to install through conda/mamba on some platforms.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/01_build_weighted_minutes.py
python src/02_build_dropoff_labels.py
python src/03_feature_engineering.py
python src/08_build_model_targets.py
python src/29_final_experiment.py
python src/30_final_interpretability.py
python src/31_downstream_interpretability_figures.py
```

Before running, read [data/README.md](data/README.md). The scripts require an authorised local copy of the raw inputs and write generated data to `data/processed/`, which is excluded from version control.

## AI collaboration disclosure

This project was developed collaboratively by the researcher and OpenAI Codex (GPT-5-based coding agent). The researcher defined the question, controlled data access, made methodological decisions and reviewed outputs. Codex helped inspect and refactor the pipeline, implement leakage controls and validation, produce diagnostics and documentation, and create presentation material. It was not used as an authority for scientific truth; final methodology and claims remain the researcher's responsibility.

The bilingual account is in [docs/AI_COLLABORATION_SUMMARY_CN_EN.md](docs/AI_COLLABORATION_SUMMARY_CN_EN.md).

## Data and ethical note

The underlying data were supplied for dissertation research and are not redistributable. This repository includes code, configuration, aggregate results and figures only. Any reuse requires independently obtained, appropriately licensed data and a review of applicable privacy, contract and competition-data restrictions.
