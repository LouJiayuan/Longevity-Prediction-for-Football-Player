# Methodology

## Analytical population

The canonical analysis includes the first and second tiers in England, Spain, Germany, Italy and France, plus the top tiers of the Netherlands, Portugal and Belgium. The modelling window is 2000-2025. Third-tier data were explored but are excluded because pre-match ClubElo coverage is not sufficient to support the match-level weighting design.

## Context-adjusted minutes

For match `j`, player `p` and season `s`, let `m_pj` be observed minutes, `E_cj` player-club Elo, `E_oj` opponent Elo and `E_ref` the seasonal median Elo of the Big Five reference leagues:

```text
Q_j = (E_cj + E_oj) / 2
w_quality,j = 1 + 0.08 * tanh((Q_j - E_ref) / 200)

g_j = E_oj - E_cj
p_j = 1 / (1 + 10^(-g_j / 400))
w_opponent,j = 1 + 0.25 * (2p_j - 1)

AM_pj = m_pj * clip(w_quality,j * w_opponent,j, 0.65, 1.35)
AM_ps = sum_j AM_pj
```

The opponent term is the standard Elo expected-score transformation. Its equivalent hyperbolic-tangent form is `2p_j - 1 = tanh((ln 10 / 800) * g_j)`. Alpha and beta are study choices, not universal constants; their stability is checked by sensitivity analysis. Club rank is retained as a model feature, not an extra minutes multiplier.

## Outcome and leakage control

For season `s`, the peak is the largest positive adjusted-minutes value strictly before `s`. A player is eligible after two historical seasons and a positive historical peak. A season is low when:

```text
adjusted_minutes_ps / prior_peak_ps < 0.65
```

The event is confirmed only if low seasons occur in two adjacent calendar seasons. The first confirmation is the only event modelled. Thresholds of 50%, 60%, 65% and 70% are audited, while 65% is primary.

Features include age, recent/current/career minute history, games played, injuries, Elo context, position groups and data-quality indicators. Future, post-event and direct peak-ratio fields are excluded. Numeric values are imputed with medians learned only from each temporal training fold. Missing future seasons end observable follow-up and are never treated as no event.

## Evaluation

Classification uses cumulative 2- and 5-year targets. Feature/model selection uses target-end years 2018-2021; locked evaluation uses target-end years 2022-2023. Probability calibration is learned from selection-period out-of-fold predictions.

Survival uses landmark rows. Feature/model selection uses 2013-2014 origins with outcomes through 2019; locked evaluation uses 2019-2020 origins with outcomes through 2025. C-index and time-dependent AUC evaluate ranking, while integrated Brier score evaluates probability error.
