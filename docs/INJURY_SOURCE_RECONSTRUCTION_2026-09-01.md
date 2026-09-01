# Injury Source Reconstruction Audit

Date: 2026-09-01

## Conclusion

The remembered source list was partly correct, but the existing
`data/injuries/injuries.csv` was not the result of those sources being joined.

Historical session evidence records this sequence:

1. The prototype first created an optional injury CSV interface.
2. The model ran with `external_injury_loaded=false`.
3. A later workspace update added `scripts/generate_synthetic_injuries.py` and a
   random CSV to bypass Kaggle authentication for pipeline testing.
4. The historical summary explicitly called it a "highly realistic synthetic
   generator" and warned that it should be replaced by real data.
5. The exact recovered code explains the current file's nine injury types,
   uniform one-to-five injuries per selected player, 2013-2024 random dates and
   impossible post-career records.

Therefore, the legacy CSV is reproducible test data, not research evidence.

## What Was Recovered

### Legacy test fixture

`scripts/00_generate_synthetic_injuries.py` preserves the original random logic
for software testing. Its default output is isolated under
`data/injuries/synthetic/` so it cannot silently replace a canonical input.

### Real public reconstruction

`scripts/00_build_public_injury_data.py` downloads or locates the two remembered
Kaggle datasets, maps identities to the structured Transfermarkt player table,
and keeps event-level and season-aggregate measurements distinct.

Identity rules are intentionally conservative:

```text
Event data:
  normalized exact name + age within one year
  or a repeated name uniquely resolved by age

Season aggregate data:
  normalized exact name + exact date of birth
```

No fuzzy match is automatically accepted. All rejected or uncertain rows remain
in audit outputs.

### Direct Transfermarkt collector

`scripts/00_collect_transfermarkt_injury_history.py` restores a player-ID-based
collector for selected cohorts. A smoke test on two player pages collected 30
event rows successfully. It is not run across the full population automatically.

### Longitudinal public extension

A third public dataset was found after the initial reconstruction:

- Kaggle: [Football Datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets)
- Documentation: [salimt/football-datasets](https://github.com/salimt/football-datasets)
- Kaggle licence field: CC0 Public Domain

It contains 143,195 Transfermarkt-ID injury-history rows for 34,561 players and
a companion profile table. `scripts/00_align_longitudinal_injuries.py` aligns
this source to all 148,140 player-seasons in the 2000-2025 modelling panel.

The alignment uses direct player IDs plus a date-of-birth audit. A one-day
tolerance is accepted because 970 of the initially conflicting event-player
identities show a systematic one-day legacy import shift. After applying that
tolerance, only 40 event rows have a larger conflict. These rows remain in an
audit file and are excluded from validated measures.

Spells are split across July-June seasons. Overlapping validated spells use the
union of calendar intervals, while 109 spells longer than 730 days remain in an
audit file but are excluded from validated days lost. No missing event is
silently converted to zero.

## Quantitative Audit

The event source contains 15,603 rows:

```text
high confidence: 15,233
medium confidence: 126
unresolved:       244
```

After source duplicate removal, 14,257 canonical events remain for 4,004
players. The aggregate source yields 1,213 high-confidence player-season rows
after identity-season deduplication.

There are 233 overlapping player-seasons in 2020. Their event-versus-aggregate
days-lost correlation is approximately 0.94, but individual discrepancies reach
more than 100 days. This supports retaining source-specific columns and rejecting
naive addition.

The longitudinal alignment adds:

```text
Panel-window event rows:               92,981
Validated event rows:                  92,832
Matched event-history players:         17,268
Panel players with any source history: 17,326
Reported event-start player-seasons:   33,965
Long-spell anomalies:                     109
Birth-date-conflict event rows:            40
```

The main limitation is historical reporting intensity, not the technical join.
Event-start prevalence among panel player-seasons is around 0.5%-1.1% in
2000-2006, 8.8%-21.9% in 2007-2015 and 29.0%-43.1% in 2016-2024. The 2025
source extract is incomplete.

## Decision for the Final Experiment

The replacement feature comparison has now been completed under the same
temporal protocol as the 28 August experiment. The old `clinical` and
`clinical_extended` sets remain cancelled. The verified ten-field injury block
is tested through `injury_only`, `age_injury`, `minutes_injury`,
`age_minutes_injury`, `age_minutes_elo_injury` and `full_model_injury`.

The locked results show no two-year gain after injury is added to a strong
model, but approximately 0.003-0.006 AUC improvement in the five-year full
classification comparisons. Survival shows almost unchanged C-index, roughly
0.002 higher five-year AUC and 0.001-0.0015 lower IBS. Injury is therefore a
secondary long-term context signal rather than a new primary model.

Keep injury excluded from the primary full-period claim because historical
reporting remains heterogeneous. Report the verified comparison as a secondary
feature-ablation result. Full results are in
`VERIFIED_INJURY_FEATURE_COMPARISON_2026-09-01.md`.
