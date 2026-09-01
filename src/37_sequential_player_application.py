"""Apply the longevity models as a sequential football decision workflow.

The two prediction questions are intentionally conditional:

1. Stable player: will the first low adjusted-minutes season of a subsequently
   confirmed decline begin within one to five years?
2. First-low player: will the next adjacent season also remain below 65% of the
   historical adjusted-minutes peak, thereby confirming the decline?

Players whose decline is already confirmed are described but not scored again.
The deployment models use only the pre-specified Age + Minutes + Elo feature set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from modeling_common import clean_feature_matrix, enforce_probability_bounds  # noqa: E402


DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "practical_application"
CONFIG_PATH = BASE_DIR / "config" / "uncertainty_experiment.json"
MAN_UTD_CLUB_ID = 985

HORIZONS = tuple(range(1, 6))
PRIORITY_HORIZON = 2
SELECTION_TARGET_YEARS = range(2018, 2022)
EVALUATION_TARGET_YEARS = range(2022, 2024)


def age_minutes_elo_features(available_columns: set[str]) -> list[str]:
    """Return the same compact feature definition used in the final experiment."""
    candidates = [
        "age",
        "age_sq",
        "final_weighted_minutes",
        "games_played",
        "previous_season_observed",
        "final_weighted_minutes_lag1",
        "final_weighted_minutes_lag2",
        "final_weighted_minutes_avg2yr",
        "final_weighted_minutes_yoy",
        "career_final_weighted_minutes",
        "career_games_played",
        "career_seasons",
        "career_adjusted_minutes_per_season",
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
    features = [column for column in candidates if column in available_columns]
    missing = sorted(set(candidates) - set(features))
    if missing:
        raise RuntimeError(f"Missing Age + Minutes + Elo features: {missing}")
    return features


def make_model() -> xgb.XGBClassifier:
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


def logit_feature(probabilities) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def fit_platt(target, raw_probabilities) -> LogisticRegression:
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(logit_feature(raw_probabilities), np.asarray(target, dtype=int))
    return calibrator


def apply_platt(calibrator, raw_probabilities) -> np.ndarray:
    return enforce_probability_bounds(
        calibrator.predict_proba(logit_feature(raw_probabilities))[:, 1]
    )


def binary_metrics(target, probability) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = enforce_probability_bounds(probability)
    return {
        "auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "n": int(len(target)),
        "event_rate": float(target.mean()),
    }


def fit_predict(train, score, features):
    X_train, _, X_score, used, medians = clean_feature_matrix(
        train, score, score, features
    )
    model = make_model()
    model.fit(X_train, train["_target"].astype(int))
    probabilities = enforce_probability_bounds(model.predict_proba(X_score)[:, 1])
    return model, probabilities, used, medians


def temporal_oof(frame, features) -> pd.DataFrame:
    parts = []
    for target_year in SELECTION_TARGET_YEARS:
        train = frame[frame["target_end_season"].lt(target_year)].copy()
        test = frame[frame["target_end_season"].eq(target_year)].copy()
        if train.empty or test.empty or train["_target"].nunique() < 2:
            continue
        _, raw, _, _ = fit_predict(train, test, features)
        parts.append(
            pd.DataFrame(
                {
                    "target": test["_target"].astype(int).to_numpy(),
                    "raw_probability": raw,
                    "target_end_season": target_year,
                }
            )
        )
    if not parts:
        raise RuntimeError("No temporal out-of-fold predictions were produced")
    return pd.concat(parts, ignore_index=True)


def evaluate_and_fit_deployment(frame, score, features, task, horizon):
    oof = temporal_oof(frame, features)
    calibrator = fit_platt(oof["target"], oof["raw_probability"])

    evaluation_train = frame[
        frame["target_end_season"].le(max(SELECTION_TARGET_YEARS))
    ].copy()
    evaluation = frame[
        frame["target_end_season"].isin(EVALUATION_TARGET_YEARS)
    ].copy()
    _, raw_evaluation, _, _ = fit_predict(evaluation_train, evaluation, features)
    calibrated_evaluation = apply_platt(calibrator, raw_evaluation)
    metrics = {
        "task": task,
        "horizon_years": int(horizon),
        "model": "xgboost_platt_calibrated",
        "feature_set": "age_minutes_elo",
        **binary_metrics(evaluation["_target"], calibrated_evaluation),
    }

    model, raw_score, used, medians = fit_predict(frame, score, features)
    calibrated_score = apply_platt(calibrator, raw_score)
    bundle = {
        "task": task,
        "horizon_years": int(horizon),
        "model": model,
        "calibrator": calibrator,
        "features": used,
        "training_medians": medians,
        "training_rows": int(len(frame)),
        "latest_training_origin_season": int(frame["season_start"].max()),
        "latest_observed_target_season": int(frame["target_end_season"].max()),
        "latest_required_followup_season": int(
            frame.get("followup_end_season", frame["target_end_season"]).max()
        ),
    }
    return calibrated_score, metrics, bundle


def build_next_season_confirmation(features: pd.DataFrame) -> pd.DataFrame:
    """Build a target only at the first low season before confirmed decline."""
    rows = []
    for player_id, group in features.groupby("player_id", sort=False):
        group = group.sort_values("season_start").reset_index(drop=True)
        prior_event = 0
        for index, current in group.iterrows():
            is_first_low = (
                int(current["dropoff_candidate"]) == 1
                and int(current["dropoff_confirmed"]) == 0
                and prior_event == 0
            )
            if is_first_low and index + 1 < len(group):
                following = group.iloc[index + 1]
                if int(following["season_start"]) == int(current["season_start"]) + 1:
                    row = current.to_dict()
                    row["_target"] = int(following["dropoff_confirmed"] == 1)
                    row["target_end_season"] = int(following["season_start"])
                    rows.append(row)
            prior_event = max(prior_event, int(current["dropoff_event"]))
    result = pd.DataFrame(rows)
    if result.empty or result["_target"].nunique() < 2:
        raise RuntimeError("Conditional confirmation target is empty or single-class")
    return result


def population_percentile(reference, values) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.searchsorted(reference, values, side="right") / len(reference)


def priority_band(percentile) -> str:
    if pd.isna(percentile):
        return "not_applicable"
    if percentile >= 0.80:
        return "high_review_priority"
    if percentile >= 0.50:
        return "medium_review_priority"
    return "routine_monitoring"


def add_monotone_early_warning_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Add monotone cumulative and non-negative yearly risk increments."""
    result = frame.copy()
    cumulative_columns = [
        f"early_warning_risk_{horizon}yr" for horizon in HORIZONS
    ]
    result[cumulative_columns] = np.maximum.accumulate(
        result[cumulative_columns].to_numpy(dtype=float), axis=1
    )
    previous = np.zeros(len(result), dtype=float)
    for horizon, column in zip(HORIZONS, cumulative_columns):
        current = result[column].to_numpy(dtype=float)
        result[f"early_warning_risk_in_year_{horizon}"] = np.clip(
            current - previous, 0.0, 1.0
        )
        previous = current
    return result


def route_and_action(row) -> str:
    if row["model_route"] == "already_confirmed_no_forecast":
        return "Observed confirmed decline: review role, availability and recovery; do not treat this as a forecast."
    if row["model_route"] == "next_season_confirmation":
        if row["review_priority"] == "high_review_priority":
            return "First low season with high persistence risk: investigate cause now and use a short review horizon."
        if row["review_priority"] == "medium_review_priority":
            return "First low season with moderate persistence risk: monitor next-season role, injuries and selection closely."
        return "First low season but lower persistence risk: treat as potentially temporary and monitor before escalation."
    if row["review_priority"] == "high_review_priority":
        return "Stable today but high onset risk: begin succession and contract scenario review; seek football-context confirmation."
    if row["review_priority"] == "medium_review_priority":
        return "Stable today with moderate onset risk: retain normal planning with closer workload and role monitoring."
    return "Stable today with lower onset risk: routine monitoring; no decline conclusion from the model."


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    data_end = int(config["data_end_season"])

    features = pd.read_parquet(DATA_DIR / "player_season_features.parquet")
    early_warning = pd.read_parquet(
        DATA_DIR / "classification_early_warning_yearly.parquet"
    )
    model_features = age_minutes_elo_features(set(features.columns))

    current = features[features["season_start"].eq(data_end)].copy()
    current["model_route"] = np.select(
        [
            current["dropoff_status"].eq("confirmed_dropoff"),
            current["dropoff_status"].eq("candidate_decline"),
        ],
        ["already_confirmed_no_forecast", "next_season_confirmation"],
        default="early_warning",
    )
    stable = current[current["model_route"].eq("early_warning")].copy()
    candidate = current[current["model_route"].eq("next_season_confirmation")].copy()

    metric_rows = []
    bundles = {}
    for horizon in HORIZONS:
        target = f"target_within_{horizon}yr"
        training = early_warning.dropna(subset=[target]).copy()
        training["_target"] = training[target].astype(int)
        training["target_end_season"] = training["season_start"].astype(int) + horizon
        # A negative early-warning label needs one season beyond the horizon
        # to verify that an onset at the boundary is not confirmed immediately
        # afterwards. Restrict deployment fitting to fully observed origin
        # cohorts so recent known positives are not retained without their
        # still-censored negative counterparts.
        training["followup_end_season"] = training["target_end_season"] + 1
        training = training[
            training["followup_end_season"].le(data_end)
        ].copy()
        probability, metrics, bundle = evaluate_and_fit_deployment(
            training,
            stable,
            model_features,
            task="early_warning",
            horizon=horizon,
        )
        stable[f"early_warning_risk_{horizon}yr"] = probability
        metric_rows.append(metrics)
        bundles[f"early_warning_{horizon}yr"] = bundle

    stable = add_monotone_early_warning_curve(stable)

    conditional = build_next_season_confirmation(features)
    confirmation_probability, metrics, bundle = evaluate_and_fit_deployment(
        conditional,
        candidate,
        model_features,
        task="next_season_confirmation",
        horizon=1,
    )
    candidate["confirmation_risk_next_season"] = confirmation_probability
    metric_rows.append(metrics)
    bundles["next_season_confirmation"] = bundle

    cumulative_columns = [
        f"early_warning_risk_{horizon}yr" for horizon in HORIZONS
    ]
    incremental_columns = [
        f"early_warning_risk_in_year_{horizon}" for horizon in HORIZONS
    ]
    current = current.merge(
        stable[["player_id", *cumulative_columns, *incremental_columns]],
        on="player_id",
        how="left",
    ).merge(
        candidate[["player_id", "confirmation_risk_next_season"]],
        on="player_id",
        how="left",
    )

    stable_reference = current.loc[
        current["model_route"].eq("early_warning"),
        f"early_warning_risk_{PRIORITY_HORIZON}yr",
    ].dropna()
    candidate_reference = current.loc[
        current["model_route"].eq("next_season_confirmation"),
        "confirmation_risk_next_season",
    ].dropna()
    current["applicable_probability"] = np.where(
        current["model_route"].eq("early_warning"),
        current[f"early_warning_risk_{PRIORITY_HORIZON}yr"],
        np.where(
            current["model_route"].eq("next_season_confirmation"),
            current["confirmation_risk_next_season"],
            np.nan,
        ),
    )
    current["risk_percentile_among_same_state"] = np.nan
    stable_mask = current["model_route"].eq("early_warning")
    candidate_mask = current["model_route"].eq("next_season_confirmation")
    percentile_columns = []
    for horizon in HORIZONS:
        risk_column = f"early_warning_risk_{horizon}yr"
        percentile_column = f"risk_percentile_{horizon}yr_among_stable"
        percentile_columns.append(percentile_column)
        current[percentile_column] = np.nan
        reference = current.loc[stable_mask, risk_column].dropna()
        current.loc[stable_mask, percentile_column] = population_percentile(
            reference, current.loc[stable_mask, risk_column]
        )
    current.loc[stable_mask, "risk_percentile_among_same_state"] = population_percentile(
        stable_reference, current.loc[stable_mask, "applicable_probability"]
    )
    current.loc[candidate_mask, "risk_percentile_among_same_state"] = population_percentile(
        candidate_reference, current.loc[candidate_mask, "applicable_probability"]
    )
    current["review_priority"] = current[
        "risk_percentile_among_same_state"
    ].map(priority_band)
    current["practical_action"] = current.apply(route_and_action, axis=1)

    output_columns = [
        "player_id",
        "name",
        "position_group",
        "season_start",
        "age",
        "final_weighted_minutes",
        "peak_adjusted_minutes",
        "adjusted_vs_peak_season",
        "dropoff_status",
        "model_route",
        *cumulative_columns,
        *incremental_columns,
        "confirmation_risk_next_season",
        *percentile_columns,
        "risk_percentile_among_same_state",
        "review_priority",
        "practical_action",
        "club_elo",
        "data_quality_score",
        "elo_imputed_minutes_share",
    ]
    population = current[output_columns].sort_values(
        ["model_route", "risk_percentile_among_same_state"],
        ascending=[True, False],
    )
    population.to_parquet(
        OUTPUT_DIR / f"sequential_population_scores_{data_end}.parquet", index=False
    )

    man_utd_ids = set(
        current.loc[current["player_club_id"].eq(MAN_UTD_CLUB_ID), "player_id"]
        .astype(int)
    )
    man_utd = population[population["player_id"].isin(man_utd_ids)].copy()
    man_utd.to_csv(
        OUTPUT_DIR / f"manchester_united_{data_end}_sequential_decisions.csv",
        index=False,
    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(OUTPUT_DIR / "sequential_model_metrics.csv", index=False)
    joblib.dump(bundles, OUTPUT_DIR / "sequential_deployment_models.joblib")

    audit = {
        "data_end_season": data_end,
        "feature_set": "age_minutes_elo",
        "features": model_features,
        "population_rows": int(len(population)),
        "stable_scored": int(stable_mask.sum()),
        "first_low_scored": int(candidate_mask.sum()),
        "already_confirmed_not_scored": int(
            current["model_route"].eq("already_confirmed_no_forecast").sum()
        ),
        "manchester_united_rows": int(len(man_utd)),
        "probability_interpretation": {
            **{
                f"early_warning_{horizon}yr": (
                    "Cumulative probability that a first low season belonging "
                    "to a subsequently confirmed decline begins within "
                    f"{horizon} year(s)."
                )
                for horizon in HORIZONS
            },
            "early_warning_risk_in_year_h": (
                "Monotone cumulative-risk difference between horizons h and "
                "h-1; this is an incremental risk mass, not a conditional hazard."
            ),
            "next_season_confirmation": (
                "Conditional probability that a player already in a first low "
                "season remains below 65% of the prior adjusted-minutes peak in "
                "the next adjacent season."
            ),
        },
        "followup_rule": (
            "Early-warning horizons are 1-5 years. Deployment training requires "
            "season_start + horizon + 1 <= data_end_season; the extra year "
            "confirms a possible onset at the fifth-year boundary."
        ),
        "priority_reference_horizon_years": PRIORITY_HORIZON,
    }
    (OUTPUT_DIR / "sequential_application_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(metrics_frame.to_string(index=False))
    print(f"\nManchester United {data_end} decisions: {len(man_utd)} players")
    print(
        man_utd[[
            "name",
            "dropoff_status",
            "model_route",
            *cumulative_columns,
            "confirmation_risk_next_season",
            "risk_percentile_among_same_state",
            "review_priority",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
