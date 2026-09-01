"""Test the incremental predictive value of Elo-adjusted minutes.

All variants predict the same official outcome: adjusted-minute, 65%,
two-adjacent-season drop-off within two or five years. The only change is the
information available to the classifier:

1. age only;
2. age plus an otherwise matched raw-minute trajectory;
3. age plus the Elo-derived adjusted-minute trajectory;
4. adjusted-minute trajectory plus aggregated Elo context.

The experiment follows the locked classification protocol used by script 29:
selection/calibration outcomes ending in 2018-2021 and an evaluation window
ending in 2022-2023. It is a post-hoc ablation of the already-inspected locked
test and must not be used to re-select the final model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from modeling_common import clean_feature_matrix, enforce_probability_bounds  # noqa: E402
from uncertainty_common import (  # noqa: E402
    add_target_end_season,
    load_uncertainty_config,
    temporal_windows,
)


KIMI_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = KIMI_DIR / "data" / "processed"
OUTPUT_DIR = KIMI_DIR / "outputs"
HORIZONS = (2, 5)
BOOTSTRAP_REPLICATES = 1000


def xgboost_classifier() -> xgb.XGBClassifier:
    """Use the locked final-experiment XGBoost hyperparameters unchanged."""
    return xgb.XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )


def add_raw_trajectory_features(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Create raw-minute analogues of the prospective adjusted trajectories."""
    ps = player_seasons.sort_values(["player_id", "season_start"]).copy()
    previous_season = ps.groupby("player_id")["season_start"].shift(1)
    two_seasons_back = ps.groupby("player_id")["season_start"].shift(2)
    adjacent_lag1 = ps["season_start"].eq(previous_season + 1)
    adjacent_lag2 = ps["season_start"].eq(two_seasons_back + 2) & adjacent_lag1

    lag1 = ps.groupby("player_id")["total_minutes"].shift(1)
    lag2 = ps.groupby("player_id")["total_minutes"].shift(2)
    ps["total_minutes_lag1"] = lag1.where(adjacent_lag1)
    ps["total_minutes_lag2"] = lag2.where(adjacent_lag2)
    ps["total_minutes_avg2yr"] = pd.concat(
        [ps["total_minutes_lag1"], ps["total_minutes_lag2"]], axis=1
    ).mean(axis=1, skipna=True)
    ps["total_minutes_yoy"] = (
        ps["total_minutes"] / ps["total_minutes_lag1"].clip(lower=1) - 1
    )
    ps["career_total_minutes"] = ps.groupby("player_id")["total_minutes"].cumsum()
    ps["career_raw_minutes_per_season"] = (
        ps["career_total_minutes"] / ps["career_seasons"].clip(lower=1)
    )
    return ps


def build_analysis_frame() -> pd.DataFrame:
    """Join official adjusted targets to raw trajectory features by landmark."""
    classification = pd.read_parquet(
        PROCESSED_DIR / "classification_confirmation_yearly.parquet"
    )
    player_seasons = pd.read_parquet(
        PROCESSED_DIR / "player_season_features.parquet"
    )
    player_seasons = add_raw_trajectory_features(player_seasons)

    if classification.duplicated(["player_id", "season_start"]).any():
        raise RuntimeError("Classification landmarks must be unique per player-season.")
    if player_seasons.duplicated(["player_id", "season_start"]).any():
        raise RuntimeError("Player-season source rows must be unique.")

    raw_columns = [
        "player_id",
        "season_start",
        "total_minutes",
        "total_minutes_lag1",
        "total_minutes_lag2",
        "total_minutes_avg2yr",
        "total_minutes_yoy",
        "career_total_minutes",
        "career_raw_minutes_per_season",
    ]
    frame = classification.merge(
        player_seasons[raw_columns],
        on=["player_id", "season_start"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(classification):
        raise RuntimeError("Raw-feature join changed the classification row count.")
    if frame[raw_columns[2:]].isna().all(axis=None):
        raise RuntimeError("Raw-minute features were not joined.")
    return frame


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return matched feature ladders; labels and test rows are held constant."""
    common = [
        "age",
        "age_sq",
        "games_played",
        "previous_season_observed",
        "career_games_played",
        "career_seasons",
    ]
    raw_trajectory = [
        "total_minutes",
        "total_minutes_lag1",
        "total_minutes_lag2",
        "total_minutes_avg2yr",
        "total_minutes_yoy",
        "career_total_minutes",
        "career_raw_minutes_per_season",
    ]
    adjusted_trajectory = [
        "final_weighted_minutes",
        "final_weighted_minutes_lag1",
        "final_weighted_minutes_lag2",
        "final_weighted_minutes_avg2yr",
        "final_weighted_minutes_yoy",
        "career_final_weighted_minutes",
        "career_adjusted_minutes_per_season",
    ]
    elo_context = [
        "club_elo",
        "club_rank_percentile",
        "avg_match_quality_elo",
        "avg_match_quality_weight",
        "avg_opponent_elo_gap",
        "avg_opponent_expected_score",
        "avg_opponent_quality_weight",
        "data_quality_score",
        "elo_imputed_minutes_share",
    ]
    sets = {
        "age_only": ["age", "age_sq"],
        "age_raw_minutes": common + raw_trajectory,
        "age_adjusted_minutes": common + adjusted_trajectory,
        "age_adjusted_minutes_elo_context": common + adjusted_trajectory + elo_context,
    }
    missing = {
        name: [column for column in columns if column not in frame.columns]
        for name, columns in sets.items()
    }
    missing = {name: columns for name, columns in missing.items() if columns}
    if missing:
        raise RuntimeError(f"Missing ablation features: {missing}")
    return sets


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Median-impute from the temporal training fold and predict test risk."""
    X_train, X_test, _, used, _ = clean_feature_matrix(
        train, test, test, columns
    )
    if not used:
        raise RuntimeError("All ablation features were constant in the training fold.")
    model = xgboost_classifier()
    model.fit(X_train, train["_target"].astype(int))
    probability = enforce_probability_bounds(model.predict_proba(X_test)[:, 1])
    return probability, used


def fit_platt(target: pd.Series, raw_probability: np.ndarray) -> LogisticRegression:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(logit, np.asarray(target, dtype=int))
    return calibrator


def apply_platt(calibrator: LogisticRegression, raw_probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    return enforce_probability_bounds(calibrator.predict_proba(logit)[:, 1])


def classification_metrics(target: pd.Series, probability: np.ndarray) -> dict[str, float]:
    target_array = np.asarray(target, dtype=int)
    probability = enforce_probability_bounds(probability)
    prevalence = float(target_array.mean())
    brier = float(brier_score_loss(target_array, probability))
    naive_brier = prevalence * (1.0 - prevalence)
    return {
        "auc": float(roc_auc_score(target_array, probability)),
        "average_precision": float(average_precision_score(target_array, probability)),
        "brier": brier,
        "brier_skill_vs_prevalence": float(1.0 - brier / naive_brier),
        "event_rate": prevalence,
    }


def selection_predictions(
    data: pd.DataFrame, columns: list[str], config: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Generate pre-evaluation temporal predictions for Platt calibration."""
    parts = []
    used_columns: list[str] = []
    for target_end in range(
        int(config["selection_target_year_start"]),
        int(config["selection_target_year_end"]) + 1,
    ):
        train = data[data["target_end_season"].lt(target_end)].copy()
        test = data[data["target_end_season"].eq(target_end)].copy()
        if train.empty or test.empty or train["_target"].nunique() < 2:
            continue
        prediction, used = fit_predict(train, test, columns)
        used_columns = used
        part = test[["player_id", "season_start", "target_end_season", "_target"]].copy()
        part["raw_probability"] = prediction
        parts.append(part)
    if not parts:
        raise RuntimeError("No pre-evaluation calibration folds were produced.")
    return pd.concat(parts, ignore_index=True), used_columns


def evaluate_variant(
    frame: pd.DataFrame,
    columns: list[str],
    variant: str,
    horizon: int,
    config: dict,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    target_column = f"target_within_{horizon}yr"
    data = frame.dropna(subset=[target_column]).copy()
    data["_target"] = data[target_column].astype(int)
    data = add_target_end_season(data, horizon)
    windows = temporal_windows(data)
    train = windows["evaluation_train"]
    test = windows["evaluation"]
    if train.empty or test.empty:
        raise RuntimeError(f"Empty evaluation data for horizon {horizon}.")
    calibration, _ = selection_predictions(data, columns, config)
    calibrator = fit_platt(calibration["_target"], calibration["raw_probability"])
    raw_probability, used = fit_predict(train, test, columns)
    probability = apply_platt(calibrator, raw_probability)
    if not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise RuntimeError("Probability bounds check failed.")

    metrics = {
        "horizon_years": horizon,
        "variant": variant,
        "model": "xgboost",
        "n_features_requested": len(columns),
        "n_features_used": len(used),
        "n_train": int(len(train)),
        "n_calibration": int(len(calibration)),
        "n_test": int(len(test)),
        **classification_metrics(test["_target"], probability),
    }
    prediction_frame = test[
        ["player_id", "season_start", "target_end_season", "_target"]
    ].rename(columns={"_target": "target"}).copy()
    prediction_frame["horizon_years"] = horizon
    prediction_frame["variant"] = variant
    prediction_frame["raw_probability"] = raw_probability
    prediction_frame["calibrated_probability"] = probability
    return metrics, prediction_frame


def clustered_bootstrap_difference(
    predictions: pd.DataFrame,
    superior: str,
    reference: str,
    horizon: int,
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> list[dict[str, float | int | str]]:
    """Player-cluster bootstrap of calibrated-probability metric differences."""
    left = predictions[
        predictions["horizon_years"].eq(horizon)
        & predictions["variant"].eq(superior)
    ][["player_id", "season_start", "target", "calibrated_probability"]].rename(
        columns={"calibrated_probability": "superior_probability"}
    )
    right = predictions[
        predictions["horizon_years"].eq(horizon)
        & predictions["variant"].eq(reference)
    ][["player_id", "season_start", "target", "calibrated_probability"]].rename(
        columns={"calibrated_probability": "reference_probability"}
    )
    merged = left.merge(
        right,
        on=["player_id", "season_start", "target"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Variants were not evaluated on identical test rows.")
    codes, players = pd.factorize(merged["player_id"], sort=False)
    n_players = len(players)
    target = merged["target"].to_numpy(dtype=int)
    superior_probability = merged["superior_probability"].to_numpy(dtype=float)
    reference_probability = merged["reference_probability"].to_numpy(dtype=float)
    rng = np.random.default_rng(42 + horizon)
    values = {"auc": [], "average_precision": [], "brier": []}

    for _ in range(n_boot):
        cluster_counts = rng.multinomial(n_players, np.full(n_players, 1.0 / n_players))
        weights = cluster_counts[codes]
        if weights[target == 1].sum() == 0 or weights[target == 0].sum() == 0:
            continue
        values["auc"].append(
            roc_auc_score(target, superior_probability, sample_weight=weights)
            - roc_auc_score(target, reference_probability, sample_weight=weights)
        )
        values["average_precision"].append(
            average_precision_score(target, superior_probability, sample_weight=weights)
            - average_precision_score(target, reference_probability, sample_weight=weights)
        )
        values["brier"].append(
            brier_score_loss(target, reference_probability, sample_weight=weights)
            - brier_score_loss(target, superior_probability, sample_weight=weights)
        )

    records = []
    for metric, differences in values.items():
        difference = np.asarray(differences, dtype=float)
        records.append({
            "horizon_years": horizon,
            "comparison": f"{superior} minus {reference}",
            "metric": metric,
            "direction": "positive_favors_superior",
            "n_test_rows": int(len(merged)),
            "n_players": int(n_players),
            "n_bootstrap": int(len(difference)),
            "mean_difference": float(difference.mean()),
            "ci_lower_95": float(np.quantile(difference, 0.025)),
            "ci_upper_95": float(np.quantile(difference, 0.975)),
            "probability_difference_le_zero": float(
                (np.sum(difference <= 0) + 1) / (len(difference) + 1)
            ),
        })
    return records


def markdown_metrics(metrics: pd.DataFrame) -> str:
    lines = [
        "| Horizon | Variant | Features | AUC | AP | Brier | Brier skill | Test rows | Event rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["horizon_years", "variant"]).itertuples():
        lines.append(
            f"| {row.horizon_years}y | {row.variant} | {row.n_features_used} | "
            f"{row.auc:.3f} | {row.average_precision:.3f} | {row.brier:.3f} | "
            f"{row.brier_skill_vs_prevalence:.3f} | {row.n_test:,} | {row.event_rate:.1%} |"
        )
    return "\n".join(lines)


def markdown_bootstrap(bootstrap: pd.DataFrame) -> str:
    lines = [
        "| Horizon | Comparison | Metric | Mean difference | 95% CI | P(diff <= 0) |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in bootstrap.sort_values(
        ["horizon_years", "comparison", "metric"]
    ).itertuples():
        lines.append(
            f"| {row.horizon_years}y | {row.comparison} | {row.metric} | "
            f"{row.mean_difference:+.3f} | [{row.ci_lower_95:+.3f}, {row.ci_upper_95:+.3f}] | "
            f"{row.probability_difference_le_zero:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_uncertainty_config()
    frame = build_analysis_frame()
    variants = feature_sets(frame)

    metrics_records = []
    prediction_parts = []
    for horizon in HORIZONS:
        for variant, columns in variants.items():
            print(f"Evaluating {horizon}y {variant}...", flush=True)
            metrics, predictions = evaluate_variant(
                frame, columns, variant, horizon, config
            )
            metrics_records.append(metrics)
            prediction_parts.append(predictions)

    metrics = pd.DataFrame(metrics_records)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_rows = metrics.groupby("horizon_years")["n_test"].nunique()
    if not expected_rows.eq(1).all():
        raise RuntimeError("Ablation variants have different locked test samples.")

    bootstrap_records = []
    comparisons = [
        ("age_raw_minutes", "age_only"),
        ("age_adjusted_minutes", "age_raw_minutes"),
        ("age_adjusted_minutes_elo_context", "age_adjusted_minutes"),
        ("age_adjusted_minutes_elo_context", "age_raw_minutes"),
    ]
    for horizon in HORIZONS:
        for superior, reference in comparisons:
            print(f"Bootstrapping {horizon}y {superior} vs {reference}...", flush=True)
            bootstrap_records.extend(clustered_bootstrap_difference(
                predictions, superior, reference, horizon
            ))
    bootstrap = pd.DataFrame(bootstrap_records)

    metrics_path = OUTPUT_DIR / "adjusted_minutes_incremental_value_metrics.csv"
    predictions_path = OUTPUT_DIR / "adjusted_minutes_incremental_value_predictions.csv"
    bootstrap_path = OUTPUT_DIR / "adjusted_minutes_incremental_value_bootstrap.csv"
    manifest_path = OUTPUT_DIR / "adjusted_minutes_incremental_value_manifest.json"
    report_path = OUTPUT_DIR / "adjusted_minutes_incremental_value_report.md"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    manifest = {
        "outcome": "official adjusted-minute 65% two-adjacent-season drop-off",
        "horizons": list(HORIZONS),
        "model": "XGBoost using final-experiment hyperparameters",
        "selection_target_end_seasons": [
            config["selection_target_year_start"],
            config["selection_target_year_end"],
        ],
        "locked_evaluation_target_end_seasons": [
            config["evaluation_target_year_start"],
            config["evaluation_target_year_end"],
        ],
        "feature_sets": variants,
        "imputation": "training-fold median only",
        "bootstrap": "1000 player-cluster resamples; positive Brier difference favors superior variant",
        "interpretation": "post-hoc locked-test ablation; not valid for model re-selection",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = f"""# Incremental Value of Elo-Adjusted Minutes

## Question

Does the Elo-derived adjusted-minute trajectory predict the **same official
adjusted-minute drop-off outcome** better than an otherwise matched raw-minute
trajectory? Does aggregated Elo context add further information beyond the
adjusted trajectory?

All variants use the same player-season landmarks, target, temporal split,
XGBoost hyperparameters, fold-only median imputation and Platt calibration.
The adjusted outcome is never replaced with a raw-minute label.

`age_adjusted_minutes` contains Elo through `final_weighted_minutes` and its
lag/career features. `age_adjusted_minutes_elo_context` additionally exposes
club, match-quality, opponent and Elo-data-quality aggregates.

## Locked evaluation results

{markdown_metrics(metrics)}

## Player-cluster bootstrap comparisons

For AUC and AP, a positive difference favors the superior variant. For Brier,
the stored difference is `reference - superior`, so a positive difference also
favors the superior variant.

{markdown_bootstrap(bootstrap)}

## Interpretation boundary

This is a downstream, post-hoc ablation on an already inspected locked test.
It quantifies the incremental value of the measurement design, but it must not
be used to choose a new final model without a future untouched temporal test.
"""
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    for path in [metrics_path, predictions_path, bootstrap_path, manifest_path, report_path]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
