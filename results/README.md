# Result snapshot

This folder contains aggregate tables from the corrected injury-free strict previous-day Elo rebuild completed on 1 September 2026. It excludes raw/processed source data, full prediction files, serialised models and player-level case exports.

## Primary outputs

- `confirmation/`: model comparison, baselines, calibration, league/tier audits and feature importance for confirmed drop-off.
- `early_warning/`: the same outputs for prediction before the first low season.
- `final_survival_model_comparison.csv`: five-model survival comparison.
- `weighted_minutes_audit.json`: strict previous-day Elo and weighting checks.
- `dropoff_threshold_sensitivity.csv`: adjusted-only 50/60/65/70% event counts.
- `alpha_beta_auc_sensitivity_metrics.csv`: complete 25-cell weight grid.
- `alpha_beta_adjusted_vs_raw_feature_gain.csv`: matched-target raw versus adjusted trajectory comparison.
- `outcome_definition_sensitivity_metrics.csv`: raw-minute and 50/60/65/70% outcome-definition checks.
- `adjusted_only_label_checks.json`: final 8,066-event label audit.

## Secondary outputs

- `verified_injury_feature_comparison/`: audited public injury-history feature
  block compared against the injury-free specification under the same temporal
  protocol, with per-task classification tables and survival deltas.
- `practical_application/`: locked evaluation metrics and the routing audit for
  the sequential early-warning / conditional-confirmation application layer.

Backward-compatible root-level `final_classification_*` aliases were removed on 1 September 2026 because they duplicated `confirmation/` byte for byte. Use the explicitly named task directories.

All files in this directory are aggregate outputs. Full predictions, held-out player examples, serialised models and source/processed datasets remain outside version control.

The same publication boundary applies to figures: aggregate calibration,
subgroup and global-importance plots are included, while player-labelled local
TreeSHAP figures are ignored.
