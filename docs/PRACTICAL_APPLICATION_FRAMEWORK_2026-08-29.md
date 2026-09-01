# Sequential Practical Application Framework

**Decision date:** 29 August 2026  
**Status:** operational interpretation layered on top of the frozen 27 August research experiment

## 1. Why the operational design changed

The 27 August headline experiment compared two- and five-year classification
endpoints for both early warning and confirmation. This remains a valid concise
research comparison, but the practical model is not restricted to those two
points. It produces a cumulative Early Warning curve at every horizon from one
to five years.
The original confirmation landmarks mix players who are still stable with
players who have already entered their first low season. Consequently, a
five-year confirmation probability combines two questions: when a low period
will begin and whether it will persist.

For practical use, these questions are separated. Long-term timing belongs to
Early Warning. Once a first low season is observed, persistence is inherently a
next-season question because the drop-off definition itself requires two
adjacent low seasons.

## 2. Outcome and state definitions

Let `AM_{p,t}` be player `p`'s adjusted minutes in season `t`, and let the
strictly prior positive peak be

```text
P_{p,t} = max(AM_{p,s}) for all observed s < t.
```

With the primary threshold `tau = 0.65`, the low-season indicator is

```text
L_{p,t} = 1[AM_{p,t} / P_{p,t} < 0.65].
```

A confirmed event requires adjacent calendar seasons:

```text
C_{p,t} = 1[L_{p,t-1} = 1 and L_{p,t} = 1 and seasons are adjacent].
```

The first low season is the onset. The second adjacent low season is the
confirmation. A player does not need to fall further in the second season; the
requirement is that adjusted minutes remain below 65% of the prior peak.

## 3. Sequential prediction tasks

| Current observed state | Applied model | Prediction |
|---|---|---|
| No current low season and no prior confirmation | Early Warning | Cumulative probability that the onset of a subsequently confirmed decline begins within 1, 2, 3, 4 and 5 years |
| Current first low season and no prior confirmation | Conditional Confirmation | Probability that the next adjacent season remains below 65%, confirming persistence |
| Prior confirmed event | No new decline forecast | The decline state is already observed; analyse role, availability and recovery |

The operational confirmation target is therefore

```text
Y_confirm(p,t+1) = L_{p,t+1}, conditional on L_{p,t}=1
                   and no confirmation by season t.
```

This is not the probability that adjusted minutes will fall again. It is the
probability that they will remain in the low region for the next adjacent
season.

For each Early Warning horizon `h = 1,...,5`, the target is

```text
Y_EW(p,t,h) = 1[confirmed-decline onset occurs in seasons t+1,...,t+h].
```

An onset in year 5 can only be confirmed by observing year 6. Therefore, the
maximum label-observation window is `5 + 1`: five forecast years plus one
confirmation year. The sixth year is not reported as a prediction horizon.

## 4. Why five-year confirmation is not deployed

Five-year confirmation retains value only as a historical benchmark showing
how the earlier, mixed-landmark formulation performs. It is excluded from the
practical interface because it:

1. overlaps with the five-year Early Warning estimand;
2. mixes onset timing with persistence;
3. conflicts with a definition based on two adjacent seasons;
4. is harder for non-technical users to translate into an action.

Long follow-up is not a task-specific reason for this decision. Both five-year
classification tasks lose recent target-complete cohorts, and Early Warning
requires an additional season after the horizon to verify a negative outcome.
This common right-censoring limitation is acknowledged rather than applied
selectively. Five-year Early Warning is retained because, from a currently
stable state, it answers a distinct strategic question about long-term onset
risk. Five-year Confirmation is excluded from deployment because, after a
first low season, persistence is structurally resolved in the next adjacent
season. The conceptual estimand, not the amount of follow-up, is the deciding
factor.

The dissertation may report the original two- and five-year confirmation
results in an appendix or model-development table. The club-facing outputs are
Early Warning at each cumulative horizon from one to five years and Conditional
Confirmation for the next adjacent season.

## 5. Practical model implementation

The deployment layer uses the compact `age_minutes_elo` feature set:

- age and age squared;
- current, lagged, recent-average and career adjusted minutes;
- games, observed career seasons and career workload;
- club Elo, club rank percentile and average match-quality context;
- opponent Elo gap and expected-score context;
- Elo imputation share and data-quality score.

XGBoost is fitted with temporal training folds. Numeric medians are learned
inside each training fold, and Platt calibration is learned from pre-evaluation
out-of-fold predictions. No prediction interval or separate uncertainty module
is added. The practical script routes every 2025 player to exactly one state,
then scores only the model appropriate to that state.

For deployment training, Early Warning origins must satisfy
`season_start + horizon + 1 <= 2025`. This applies the same complete-follow-up
principle to positive and negative rows and prevents recent known positives
from entering without their still-censored negative counterparts. The latest
fully observed origins become progressively older as the horizon expands:

| Horizon | Deployment rows | Latest fully observed origin | Required observation end |
|---:|---:|---:|---:|
| 1 year | 53,610 | 2023 | 2025 |
| 2 years | 44,594 | 2022 | 2025 |
| 3 years | 39,244 | 2021 | 2025 |
| 4 years | 35,764 | 2020 | 2025 |
| 5 years | 33,123 | 2019 | 2025 |

The five-year estimate should be presented as a long-term planning signal
because it is learned from an older cohort and has a high cumulative event
rate. The survival analysis is retained as a censoring-aware companion for
long-horizon evidence, although it does not remove the need to disclose cohort
age and model assumptions.

The five separately calibrated cumulative probabilities are projected to be
non-decreasing across horizons. Their adjacent differences are also reported
as `risk_in_year_1` to `risk_in_year_5`. These differences are incremental risk
mass, not conditional hazards. The default review-priority band uses the
two-year percentile for near-term planning, while all five horizon-specific
percentiles remain available in the output.

## 6. Locked practical diagnostics

| Operational task | Horizon | AUC | Average precision | Brier | Test event rate |
|---|---:|---:|---:|---:|---:|
| Early Warning | 1 year | 0.742 | 0.295 | 0.113 | 14.6% |
| Early Warning | 2 years | 0.673 | 0.452 | 0.196 | 30.6% |
| Early Warning | 3 years | 0.677 | 0.619 | 0.225 | 45.1% |
| Early Warning | 4 years | 0.684 | 0.747 | 0.219 | 58.8% |
| Early Warning | 5 years | 0.689 | 0.836 | 0.187 | 71.3% |
| Conditional Confirmation | next season | 0.654 | 0.722 | 0.226 | 59.1% |

The models provide a screening and review order, not an automated employment
decision. The conditional confirmation model has moderate discrimination and
must be combined with football information about injuries, tactical role,
transfers and manager selection. Five-year probabilities are cumulative and
have a high base rate; they must not be interpreted as an exact predicted year
of decline.

## 7. Club case examples

The routed club case study is withheld from this public repository. Named
player-level probabilities and the Manchester United-supplied source data fall
outside the publication boundary stated in the repository README. The routing
logic, locked diagnostics and interpretation rules in sections 3 to 6 are
complete without them.

## 8. Reproducible outputs

- `src/37_sequential_player_application.py`: state routing, temporal model
  fitting, calibration and scoring.
- `results/practical_application/sequential_model_metrics.csv`: locked
  evaluation metrics.
- `results/practical_application/sequential_application_audit.json`: feature,
  interpretation and routing audit.

Named-player score tables (`manchester_united_2025_sequential_decisions.csv`),
the full 2025 population scores and the fitted deployment models are restricted
and remain outside this repository.

## 9. Professor-ready explanation

Following our discussion, I reconsidered the distinction between early warning
and confirmation. For players who have not yet entered a low-minutes season,
Early Warning estimates a cumulative onset curve for each of the next one to
five years. Once a player records a first low season,
defined as adjusted minutes below 65% of the previous peak, confirmation is
inherently a next-season question: whether the following adjacent season
remains below the same threshold. A five-year confirmation model mixes onset
timing with persistence and overlaps with Early Warning. You are also correct
that limited five-year follow-up affects Early Warning as well, so I should not
use censoring selectively to justify removing only five-year Confirmation.
Instead, I estimate Early Warning at every cumulative horizon from one to five
years because each horizon answers a distinct timing question for currently
stable players. A possible onset in year 5 requires year 6 only to verify
persistence; year 6 is not itself a prediction horizon. The operational
framework therefore uses a one- to five-year Early Warning curve but
next-season Conditional Confirmation because of the estimand structure, not
because only Confirmation has a follow-up problem. The former two- and
five-year confirmation results remain methodological benchmarks rather than
club-facing outputs.
