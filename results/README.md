# Result snapshot

This folder contains small aggregate tables from the final run on 24 August 2026. It excludes player-season source data, full prediction files and serialised model objects.

- `final_experiment_manifest.json`: frozen experiment definition and sample counts.
- `final_classification_vertical_comparison.csv`: naive baselines and feature-set comparison.
- `final_classification_model_comparison.csv`: five-model classification comparison.
- `final_survival_model_comparison.csv`: five-model locked survival comparison.
- `final_classification_cluster_bootstrap.csv`: clustered uncertainty for baseline differences.
- `final_xgboost_permutation_importance.csv`: held-out permutation importance.
