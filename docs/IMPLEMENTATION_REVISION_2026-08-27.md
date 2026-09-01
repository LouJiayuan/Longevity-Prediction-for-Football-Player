# Final Method Revision: 27 August 2026

> **Historical snapshot:** the classification feature sets and results below
> were superseded on 1 September 2026 after the injury-source audit. Do not cite
> its `clinical_extended` or all-XGBoost results. Use `METHODOLOGY.md` and the
> current files under `results/<task>/`.

## Decisions implemented

1. ClubElo is now the latest observation from a calendar day strictly before the match. The code normalizes both dates, uses `searchsorted(..., side="left") - 1`, records same-day availability for audit and asserts a minimum one-day lag.
2. Classification is split into two estimands. `confirmation` predicts the future second low season and allows a current first-low-season candidate. `early_warning` excludes those candidates, predicts the future first low season only when it is subsequently confirmed, and requires an extra observed season before assigning a negative.
3. The official label remains adjusted minutes below 65% of the strictly prior positive peak for two adjacent seasons. Raw minutes and 50/60/70% thresholds are sensitivity analyses only.
4. No additional uncertainty module is added. Existing calibration and aggregate evaluation tables remain available.

## Rebuilt data

| Item | Result |
|---|---:|
| Player-match rows | 3,146,714 |
| Matches | 113,890 |
| Players | 34,089 |
| Player-seasons | 148,140 |
| First confirmed events | 8,066 |
| Confirmation landmarks | 85,135 |
| Early-warning landmarks | 55,829 |
| Candidate landmarks in early warning | 0 |
| Same-day Elo values excluded | 228,748 club-sides |
| Minimum direct Elo lag | 1 calendar day |
| Direct player-club Elo coverage | 98.83% |
| Direct opponent Elo coverage | 98.83% |

## Locked classification results

| Task | Horizon | Selected features | Selected model | AUC | AP | Brier |
|---|---:|---|---|---:|---:|---:|
| Confirmation | 2 years | clinical extended | XGBoost | 0.811 | 0.579 | 0.138 |
| Confirmation | 5 years | clinical extended | XGBoost | 0.736 | 0.849 | 0.189 |
| Early warning | 2 years | full model | XGBoost | 0.675 | 0.457 | 0.195 |
| Early warning | 5 years | full model | XGBoost | 0.693 | 0.845 | 0.186 |

The stricter early-warning task is materially harder. It should be the primary result when making a claim about prediction before decline is visible; confirmation is a useful operational monitoring task.

## Robustness result

Across 25 combinations of `alpha=0.15-0.35` and `beta=0.04-0.12`, 2-year AUC ranges from 0.805 to 0.810 and 5-year AUC from 0.713 to 0.727. The maximum event-row flip rate is 0.41%. These ranges support stability and must not be used to choose the parameter with the highest test AUC.

## Verification

Core tests cover strict previous-day Elo, separation of the two targets and the extra follow-up season required for early-warning negatives. Both the development workspace and GitHub package pass all three tests.

The GitHub snapshot also records the exact locked dependency versions. For exact interpretation of LightGBM, note that `subsample=0.85` is stored while the library default `subsample_freq=0` leaves row bagging inactive; no post-test hyperparameter change or rerun was made.
