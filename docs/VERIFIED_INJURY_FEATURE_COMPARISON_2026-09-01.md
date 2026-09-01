# Verified Injury Feature Comparison

Date: 2026-09-01

## Purpose

This experiment replaces the invalid legacy `clinical` and
`clinical_extended` feature sets used in the 28 August presentation. Those
sets depended on a synthetic injury fixture and are cancelled as research
evidence.

The replacement retains the same vertical comparison logic while using the
audited Transfermarkt-ID public injury-history alignment. The outcome remains
the adjusted-minutes definition: below 65% of the strictly prior positive peak
for two adjacent seasons. Injury does not define the outcome.

## Verified Injury Block

The replacement injury block contains ten prospective player-season fields:

1. current validated unique days lost;
2. current reported absence count;
3. current severe-absence count;
4. current major-absence count;
5. current reported games missed;
6. previous-season days lost;
7. two-season days lost;
8. two-season absence count;
9. career cumulative days lost through the current season;
10. source-profile availability indicator.

No `history_zero_assumption` field is used. That assumption depends on whether
a player has an injury at any later point in the source and would leak future
information into earlier landmarks. The experiment uses only the explicit
profile-coverage assumption, while remaining missing values are imputed inside
each temporal training fold.

## Feature Sets

The old clinical labels are removed. The comparison contains:

- `age_only` and `minutes_only` naive baselines;
- the existing `age_career`, `age_plus_minutes`, `age_minutes_elo` and
  `full_model` specifications;
- `injury_only`;
- `age_injury` and `minutes_injury`;
- `age_minutes_injury`;
- `age_minutes_elo_injury`;
- `full_model_injury`.

Classification uses the same 2018-2021 target-end selection folds and locked
2022-2023 evaluation as the corrected final experiment. Logistic Regression,
Random Forest, Gradient Boosting, XGBoost and LightGBM are compared. Survival
retains the same Cox-based feature selection, five survival algorithms and
locked 2019-2020 origin cohort.

Every vertical-comparison row uses the algorithm chosen from the selection
period for that feature set. Consequently, a descriptive `full_model_injury`
row is not automatically the primary final model, even when it has the highest
locked AUC at one horizon.

## Classification Results

### Compact Age + Minutes + Elo specification

| Task | Horizon | Without injury AUC | With injury AUC | AUC difference | Brier difference |
|---|---:|---:|---:|---:|---:|
| Confirmation | 2 years | 0.8112 | 0.8107 | -0.0005 | +0.0005 |
| Confirmation | 5 years | 0.7360 | 0.7372 | +0.0012 | -0.0002 |
| Early Warning | 2 years | 0.6686 | 0.6670 | -0.0016 | +0.0007 |
| Early Warning | 5 years | 0.6861 | 0.6922 | +0.0061 | -0.0009 |

### Full specification

| Task | Horizon | Without injury AUC | With injury AUC | AUC difference | Brier difference |
|---|---:|---:|---:|---:|---:|
| Confirmation | 2 years | 0.8116 | 0.8105 | -0.0011 | +0.0010 |
| Confirmation | 5 years | 0.7381 | 0.7410 | +0.0029 | -0.0001 |
| Early Warning | 2 years | 0.6723 | 0.6713 | -0.0010 | +0.0010 |
| Early Warning | 5 years | 0.6917 | 0.6972 | +0.0055 | -0.0009 |

Feature selection chooses `age_minutes_elo_injury` for Confirmation because it
is within 0.005 AUC of the best selection result while using 32 rather than 63
features. Early Warning retains the injury-free `full_model`; although
`full_model_injury` has a slightly higher selection mean AUC, the smaller full
model is within the pre-specified 0.005 tolerance.

Injury alone is not a strong model. Its locked AUC is 0.630/0.600 for
Confirmation and 0.574/0.583 for Early Warning at two/five years. The injury
block is therefore a secondary context signal, not a replacement for age,
playing-time trajectory or Elo context.

## Survival Results

Cox vertical comparison gives:

| Feature set | C-index | AUC 2y | AUC 5y | IBS 1-5y |
|---|---:|---:|---:|---:|
| Age + Minutes + Elo | 0.7304 | 0.7736 | 0.6904 | 0.2287 |
| Age + Minutes + Elo + Injury | 0.7297 | 0.7728 | 0.6921 | 0.2276 |
| Full model | 0.7275 | 0.7700 | 0.6844 | 0.2309 |
| Full model + Injury | 0.7265 | 0.7686 | 0.6862 | 0.2295 |

Adding injury changes C-index by less than 0.001 and slightly reduces 2-year
AUC, but improves 5-year AUC by approximately 0.002 and lowers IBS by
0.001-0.0015. The survival feature-selection rule chooses
`age_minutes_elo_injury`. Random Survival Forest is selected during the model
selection stage; in locked evaluation it has the lowest IBS (0.216), while
Gradient Boosting Survival has the highest C-index (0.748) and 2-year AUC
(0.783). This is the same discrimination-versus-calibration trade-off already
seen in the injury-free experiment.

## Conclusion for the Dissertation

The defensible conclusion is not that injury history transforms prediction.
The verified public injury proxy adds little or no information at the two-year
horizon and a small amount of information at the five-year horizon. Its main
value is as a long-term contextual supplement.

The primary full-period claim should continue to use the injury-free model.
The verified injury extension can be reported as a secondary feature-ablation
result showing that the main Age + Minutes + Elo findings are not dependent on
the old synthetic injury fields.

Injury histories are public availability-reporting proxies, not medical
records. Reporting is sparse before 2007 and changes over time. These results
must not be described as evidence from complete clinical data.
