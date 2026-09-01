"""Final two-horizon longevity experiment.

The script implements the dissertation experiment in two stages:

1. Select one feature set using pre-evaluation temporal data only.
2. Compare five model families horizontally on locked 2- and 5-year tests.

The operational outcome is the adjusted-minutes-only, 65%, two-adjacent-season
drop-off event produced by ``02_build_dropoff_labels.py``. Numeric imputation is
estimated inside each training fold. No target, peak-relative, future,
post-event or non-empirical injury field is admitted to a feature set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import WeibullAFTFitter
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from modeling_common import (  # noqa: E402
    MODEL_DIR,
    OUT_DIR,
    base_feature_columns,
    clean_feature_matrix,
    enforce_probability_bounds,
)
from uncertainty_common import (  # noqa: E402
    add_target_end_season,
    load_uncertainty_config,
    temporal_windows,
)

warnings.filterwarnings("ignore")

HORIZONS = (2, 5)
CLASSIFICATION_TASK_FILES = {
    "confirmation": "classification_confirmation_yearly.parquet",
    "early_warning": "classification_early_warning_yearly.parquet",
}
CLASSIFICATION_MODELS = (
    "logistic",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
)
SURVIVAL_MODELS = (
    "cox_ph",
    "weibull_aft",
    "random_survival_forest",
    "extra_survival_trees",
    "gradient_boosting_survival",
)
FEATURE_SELECTION_TOLERANCE = 0.005
SURVIVAL_SELECTION_TRAIN_CUTOFF = 2013
SURVIVAL_SELECTION_ORIGIN_START = 2013
SURVIVAL_SELECTION_ORIGIN_END = 2014
SURVIVAL_SELECTION_OUTCOME_CUTOFF = 2019
SURVIVAL_EVALUATION_TRAIN_CUTOFF = 2019
SURVIVAL_EVALUATION_OUTCOME_CUTOFF = 2025


def unique_available(candidates, available):
    return list(dict.fromkeys(c for c in candidates if c in available))


def define_feature_sets(all_features):
    """Return the pre-specified vertical feature-set ladder."""
    available = set(all_features)
    age = unique_available(["age"], available)
    age_nonlinear = unique_available(["age", "age_sq"], available)
    minute_baseline = unique_available(["final_weighted_minutes"], available)
    minute_history = unique_available([
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
    ], available)
    elo_context = unique_available([
        "club_elo",
        "club_rank_percentile",
        "avg_match_quality_elo",
        "avg_match_quality_weight",
        "avg_opponent_elo_gap",
        "avg_opponent_expected_score",
        "avg_opponent_quality_weight",
        "data_quality_score",
        "elo_imputed_minutes_share",
    ], available)
    sets = {
        "age_only": age,
        "minutes_only": minute_baseline,
        "age_career": unique_available([
            "age", "age_sq", "career_final_weighted_minutes", "career_seasons"
        ], available),
        "age_plus_minutes": unique_available(
            age_nonlinear + minute_history, available
        ),
        "age_minutes_elo": unique_available(
            age_nonlinear + minute_history + elo_context, available
        ),
        "full_model": list(all_features),
    }
    for name, columns in sets.items():
        if not columns:
            raise RuntimeError(f"Feature set {name} is empty")
        forbidden = [
            c for c in columns
            if c.startswith(("target_", "dropoff_", "legacy_dropoff_"))
            or c in {
                "peak_season", "peak_adjusted_minutes",
                "adjusted_vs_peak_season", "final_vs_peak",
                "recovery_after_dropoff", "injury_associated_dropoff",
            }
        ]
        if forbidden:
            raise RuntimeError(f"Leakage fields in {name}: {forbidden}")
        excluded_injury = [c for c in columns if c.startswith("injury_")]
        if excluded_injury:
            raise RuntimeError(
                f"Excluded injury fields in {name}: {excluded_injury}"
            )
    return sets


def classification_model(name):
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, random_state=42)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=9,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=4,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            min_samples_leaf=15,
            random_state=42,
        )
    if name == "xgboost":
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
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=25,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
    raise KeyError(name)


def logit_feature(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(y, raw_probabilities):
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(logit_feature(raw_probabilities), np.asarray(y, dtype=int))
    return calibrator


def apply_platt(calibrator, raw_probabilities):
    return enforce_probability_bounds(
        calibrator.predict_proba(logit_feature(raw_probabilities))[:, 1]
    )


def binary_metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = enforce_probability_bounds(probabilities)
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def fit_classification(train, test, features, model_name):
    X_train, X_test, _, used, medians = clean_feature_matrix(
        train, test, test, features
    )
    model = classification_model(model_name)
    model.fit(X_train, train["_target"].astype(int))
    predictions = enforce_probability_bounds(
        model.predict_proba(X_test)[:, 1]
    )
    return model, predictions, used, medians


def classification_selection(frame, feature_sets, config):
    records = []
    oof = defaultdict(list)
    for horizon in HORIZONS:
        target = f"target_within_{horizon}yr"
        data = frame.dropna(subset=[target]).copy()
        data["_target"] = data[target].astype(int)
        data = add_target_end_season(data, horizon)
        for target_year in range(
            config["selection_target_year_start"],
            config["selection_target_year_end"] + 1,
        ):
            train = data[data["target_end_season"].lt(target_year)].copy()
            test = data[data["target_end_season"].eq(target_year)].copy()
            if train.empty or test.empty or train["_target"].nunique() < 2:
                continue
            print(
                f"  selection horizon={horizon} target_end={target_year} "
                f"train={len(train)} test={len(test)}",
                flush=True,
            )
            for feature_name, features in feature_sets.items():
                for model_name in CLASSIFICATION_MODELS:
                    _, pred, used, _ = fit_classification(
                        train, test, features, model_name
                    )
                    key = (horizon, feature_name, model_name)
                    oof[key].append(pd.DataFrame({
                        "player_id": test["player_id"].to_numpy(),
                        "season_start": test["season_start"].to_numpy(),
                        "target_end_season": test["target_end_season"].to_numpy(),
                        "target": test["_target"].to_numpy(),
                        "raw_probability": pred,
                    }))
                    records.append({
                        "horizon_years": horizon,
                        "target_end_season": target_year,
                        "feature_set": feature_name,
                        "model": model_name,
                        "n_features": len(used),
                        **binary_metrics(test["_target"], pred),
                        "n": len(test),
                    })
    if not records:
        raise RuntimeError("No classification selection folds were produced")
    return pd.DataFrame(records), {
        key: pd.concat(parts, ignore_index=True) for key, parts in oof.items()
    }


def save_classification_selection_cache(
    metrics, oof_predictions, output_dir, signature
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "final_classification_selection_folds.csv", index=False
    )
    parts = []
    for (horizon, feature_set, model), frame in oof_predictions.items():
        out = frame.copy()
        out["horizon_years"] = horizon
        out["feature_set"] = feature_set
        out["model"] = model
        parts.append(out)
    pd.concat(parts, ignore_index=True).to_parquet(
        output_dir / "final_classification_selection_oof.parquet", index=False
    )
    (output_dir / "final_classification_selection_cache.json").write_text(
        json.dumps({"signature": signature}, indent=2), encoding="utf-8"
    )


def load_classification_selection_cache(output_dir, expected_signature):
    metrics_path = output_dir / "final_classification_selection_folds.csv"
    oof_path = output_dir / "final_classification_selection_oof.parquet"
    metadata_path = output_dir / "final_classification_selection_cache.json"
    if not metrics_path.exists() or not oof_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != expected_signature:
        return None
    metrics = pd.read_csv(metrics_path)
    stored = pd.read_parquet(oof_path)
    oof = {}
    for key, frame in stored.groupby(
        ["horizon_years", "feature_set", "model"], sort=False
    ):
        oof[(int(key[0]), str(key[1]), str(key[2]))] = frame.drop(
            columns=["horizon_years", "feature_set", "model"]
        ).reset_index(drop=True)
    return metrics, oof


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_cache_signature(data_path, config, feature_sets):
    payload = {
        "data_sha256": file_sha256(data_path),
        "selection_window": {
            key: config[key]
            for key in (
                "data_end_season",
                "selection_target_year_start",
                "selection_target_year_end",
                "evaluation_target_year_start",
                "evaluation_target_year_end",
            )
        },
        "horizons": list(HORIZONS),
        "models": list(CLASSIFICATION_MODELS),
        "feature_sets": feature_sets,
        "selection_tolerance": FEATURE_SELECTION_TOLERANCE,
        "cache_schema": 2,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def choose_feature_set(selection_metrics):
    summary = (
        selection_metrics.groupby("feature_set", as_index=False)
        .agg(
            mean_selection_auc=("auc", "mean"),
            minimum_selection_auc=("auc", "min"),
            n_features=("n_features", "max"),
        )
    )
    best = float(summary["mean_selection_auc"].max())
    eligible = summary[
        summary["mean_selection_auc"].ge(best - FEATURE_SELECTION_TOLERANCE)
    ].copy()
    chosen = eligible.sort_values(
        ["n_features", "mean_selection_auc"], ascending=[True, False]
    ).iloc[0]["feature_set"]
    summary["within_selection_tolerance"] = summary[
        "mean_selection_auc"
    ].ge(best - FEATURE_SELECTION_TOLERANCE)
    summary["selected"] = summary["feature_set"].eq(chosen)
    return str(chosen), summary


def choose_model_for_feature(selection_metrics, feature_name):
    scores = (
        selection_metrics[selection_metrics["feature_set"].eq(feature_name)]
        .groupby("model", as_index=False)
        .agg(mean_selection_auc=("auc", "mean"))
        .sort_values(["mean_selection_auc", "model"], ascending=[False, True])
    )
    return str(scores.iloc[0]["model"])


def classification_evaluation(
    frame, feature_sets, selection_metrics, oof_predictions, selected_feature
):
    vertical_records = []
    horizontal_records = []
    prediction_records = []
    selected_models = {}

    vertical_models = {
        name: choose_model_for_feature(selection_metrics, name)
        for name in feature_sets
    }

    for horizon in HORIZONS:
        target = f"target_within_{horizon}yr"
        data = frame.dropna(subset=[target]).copy()
        data["_target"] = data[target].astype(int)
        data = add_target_end_season(data, horizon)
        windows = temporal_windows(data)
        train = windows["evaluation_train"]
        test = windows["evaluation"]
        if train.empty or test.empty:
            raise RuntimeError(f"Empty classification evaluation horizon {horizon}")

        for feature_name, model_name in vertical_models.items():
            key = (horizon, feature_name, model_name)
            oof = oof_predictions[key]
            calibrator = fit_platt(oof["target"], oof["raw_probability"])
            _, raw, used, _ = fit_classification(
                train, test, feature_sets[feature_name], model_name
            )
            calibrated = apply_platt(calibrator, raw)
            vertical_records.append({
                "horizon_years": horizon,
                "feature_set": feature_name,
                "selected_model_from_selection": model_name,
                "n_features": len(used),
                **binary_metrics(test["_target"], calibrated),
                "n": len(test),
                "event_rate": float(test["_target"].mean()),
            })

        for model_name in CLASSIFICATION_MODELS:
            key = (horizon, selected_feature, model_name)
            oof = oof_predictions[key]
            calibrator = fit_platt(oof["target"], oof["raw_probability"])
            _, raw, used, _ = fit_classification(
                train, test, feature_sets[selected_feature], model_name
            )
            calibrated = apply_platt(calibrator, raw)
            horizontal_records.append({
                "horizon_years": horizon,
                "feature_set": selected_feature,
                "model": model_name,
                "n_features": len(used),
                **binary_metrics(test["_target"], calibrated),
                "n": len(test),
                "event_rate": float(test["_target"].mean()),
            })
            prediction_frame = test[[
                "player_id", "season_start", "target_end_season", "_target"
            ]].rename(columns={"_target": "target"}).copy()
            prediction_frame["horizon_years"] = horizon
            prediction_frame["feature_set"] = selected_feature
            prediction_frame["model"] = model_name
            prediction_frame["raw_probability"] = raw
            prediction_frame["calibrated_probability"] = calibrated
            prediction_records.extend(prediction_frame.to_dict("records"))

        selected_models[horizon] = choose_model_for_feature(
            selection_metrics[
                selection_metrics["horizon_years"].eq(horizon)
            ],
            selected_feature,
        )

    return (
        pd.DataFrame(vertical_records),
        pd.DataFrame(horizontal_records),
        pd.DataFrame(prediction_records),
        selected_models,
    )


def cluster_bootstrap_classification(
    vertical, predictions, selection_metrics, selected_feature, n_boot=1000
):
    records = []
    rng = np.random.default_rng(42)
    for horizon in HORIZONS:
        final_model = choose_model_for_feature(
            selection_metrics[
                selection_metrics["horizon_years"].eq(horizon)
            ],
            selected_feature,
        )
        comparisons = {
            "age_only": choose_model_for_feature(
                selection_metrics[
                    selection_metrics["horizon_years"].eq(horizon)
                ],
                "age_only",
            ),
            "minutes_only": choose_model_for_feature(
                selection_metrics[
                    selection_metrics["horizon_years"].eq(horizon)
                ],
                "minutes_only",
            ),
        }
        final = predictions[
            predictions["horizon_years"].eq(horizon)
            & predictions["model"].eq(final_model)
        ][["player_id", "season_start", "target", "calibrated_probability"]].rename(
            columns={"calibrated_probability": "final_probability"}
        )

        for baseline_feature, baseline_model in comparisons.items():
            # Reconstruct evaluation probabilities from the vertical metrics run
            # by fitting the pre-selected baseline once more is avoided here;
            # baseline predictions are generated below from the stored vertical
            # model only when it is also in the selected horizontal feature set.
            # The caller supplies them through a dedicated prediction table.
            baseline_key = f"{baseline_feature}:{baseline_model}"
            baseline_rows = vertical.attrs.get("prediction_tables", {}).get(
                (horizon, baseline_feature, baseline_model)
            )
            if baseline_rows is None:
                continue
            merged = final.merge(
                baseline_rows,
                on=["player_id", "season_start", "target"],
                how="inner",
            )
            player_codes, players = pd.factorize(
                merged["player_id"], sort=False
            )
            n_players = len(players)
            deltas = []
            for _ in range(n_boot):
                cluster_counts = rng.multinomial(
                    n_players, np.full(n_players, 1.0 / n_players)
                )
                row_weights = cluster_counts[player_codes]
                positive_weight = row_weights[merged["target"].eq(1)].sum()
                negative_weight = row_weights[merged["target"].eq(0)].sum()
                if positive_weight == 0 or negative_weight == 0:
                    continue
                deltas.append(
                    roc_auc_score(
                        merged["target"],
                        merged["final_probability"],
                        sample_weight=row_weights,
                    )
                    - roc_auc_score(
                        merged["target"],
                        merged["baseline_probability"],
                        sample_weight=row_weights,
                    )
                )
            values = np.asarray(deltas, dtype=float)
            records.append({
                "horizon_years": horizon,
                "comparison": f"{selected_feature}:{final_model} minus {baseline_key}",
                "n_players": n_players,
                "mean_auc_difference": float(values.mean()),
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "bootstrap_probability_difference_le_zero": float(
                    (np.sum(values <= 0) + 1) / (len(values) + 1)
                ),
            })
    return pd.DataFrame(records)


def to_surv(frame):
    return Surv.from_arrays(
        event=frame["event"].astype(bool).to_numpy(),
        time=frame["time_to_event"].astype(float).to_numpy(),
    )


def administrative_training(frame, cutoff):
    train = frame[frame["season_start"].lt(cutoff)].copy()
    event_season = train["season_start"] + train["time_to_event"]
    observed_event = train["event"].eq(1) & event_season.le(cutoff)
    train["time_to_event"] = np.where(
        observed_event,
        train["time_to_event"],
        np.minimum(
            train["observed_future_seasons"],
            cutoff - train["season_start"],
        ),
    ).astype(int)
    train["event"] = observed_event.astype(int)
    return train[train["time_to_event"].ge(1)].copy()


def administrative_test_cohort(frame, origin_start, origin_end, outcome_cutoff):
    test = frame[
        frame["season_start"].between(origin_start, origin_end)
    ].copy()
    event_season = test["season_start"] + test["time_to_event"]
    observed_event = test["event"].eq(1) & event_season.le(outcome_cutoff)
    maximum_followup = outcome_cutoff - test["season_start"]
    test["time_to_event"] = np.where(
        observed_event,
        test["time_to_event"],
        np.minimum(test["observed_future_seasons"], maximum_followup),
    ).astype(int)
    test["event"] = observed_event.astype(int)
    return test[test["time_to_event"].ge(1)].copy()


def survival_model(name, selection=False):
    trees = 60 if selection else 150
    if name == "cox_ph":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", CoxPHSurvivalAnalysis(alpha=0.1)),
        ])
    if name == "weibull_aft":
        return WeibullAFTFitter(penalizer=0.1)
    if name == "random_survival_forest":
        return RandomSurvivalForest(
            n_estimators=trees,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=35,
            max_features="sqrt",
            random_state=42,
            n_jobs=4,
        )
    if name == "extra_survival_trees":
        return ExtraSurvivalTrees(
            n_estimators=trees,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=35,
            max_features="sqrt",
            random_state=42,
            n_jobs=4,
        )
    if name == "gradient_boosting_survival":
        return GradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=80 if not selection else 50,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=25,
            random_state=42,
        )
    raise KeyError(name)


def fit_survival(train, test, features, model_name, selection=False):
    X_train, X_test, _, used, medians = clean_feature_matrix(
        train, test, test, features
    )
    y_train = to_surv(train)
    if model_name == "weibull_aft":
        scaler = StandardScaler()
        train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=used, index=X_train.index
        )
        test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=used, index=X_test.index
        )
        fit_frame = train_scaled.copy()
        fit_frame["time_to_event"] = train["time_to_event"].to_numpy()
        fit_frame["event"] = train["event"].to_numpy()
        model = survival_model(model_name, selection=selection)
        model.fit(fit_frame, duration_col="time_to_event", event_col="event")
        fitted = (model, scaler)
        return fitted, X_train, X_test, train_scaled, test_scaled, used, medians
    model = survival_model(model_name, selection=selection)
    model.fit(X_train, y_train)
    return model, X_train, X_test, X_train, X_test, used, medians


def survival_predictions(fitted, model_name, X_for_model, times):
    times = np.asarray(times, dtype=float)
    if model_name == "weibull_aft":
        model, _ = fitted
        matrix = model.predict_survival_function(
            X_for_model, times=times
        ).T.to_numpy()
        risk = -model.predict_expectation(X_for_model).to_numpy()
        return risk, enforce_probability_bounds(matrix)

    model = fitted
    if model_name == "cox_ph":
        estimator = model.named_steps["model"]
        transformed = model.named_steps["scaler"].transform(X_for_model)
        functions = estimator.predict_survival_function(transformed)
        risk = model.predict(X_for_model)
    else:
        functions = model.predict_survival_function(X_for_model)
        risk = model.predict(X_for_model)
    matrix = np.asarray([
        [float(fn(t)) for t in times] for fn in functions
    ])
    return np.asarray(risk, dtype=float), enforce_probability_bounds(matrix)


def survival_metrics(train, test, risk, survival_matrix, times=(1, 2, 3, 4, 5)):
    y_train = to_surv(train)
    y_test = to_surv(test)
    result = {
        "c_index": float(concordance_index_censored(
            test["event"].astype(bool).to_numpy(),
            test["time_to_event"].astype(float).to_numpy(),
            np.asarray(risk, dtype=float),
        )[0])
    }
    times = np.asarray(times, dtype=float)
    for horizon in HORIZONS:
        try:
            _, auc = cumulative_dynamic_auc(
                y_train, y_test, np.asarray(risk, dtype=float),
                np.asarray([float(horizon)]),
            )
            result[f"auc_{horizon}yr"] = float(auc)
        except Exception:
            result[f"auc_{horizon}yr"] = np.nan
    try:
        result["integrated_brier_score_1to5"] = float(
            integrated_brier_score(y_train, y_test, survival_matrix, times)
        )
    except Exception:
        result["integrated_brier_score_1to5"] = np.nan
    return result


def survival_selection(frame, feature_sets):
    """Select features with a fixed, interpretable Cox PH reference model."""
    train = administrative_training(
        frame, SURVIVAL_SELECTION_TRAIN_CUTOFF
    )
    test = administrative_test_cohort(
        frame,
        SURVIVAL_SELECTION_ORIGIN_START,
        SURVIVAL_SELECTION_ORIGIN_END,
        SURVIVAL_SELECTION_OUTCOME_CUTOFF,
    )
    records = []
    for feature_name, features in feature_sets.items():
        model_name = "cox_ph"
        print(f"  survival feature selection: {feature_name}", flush=True)
        try:
            fitted, _, _, _, X_test_model, used, _ = fit_survival(
                train, test, features, model_name, selection=True
            )
            risk, surv = survival_predictions(
                fitted, model_name, X_test_model, [1, 2, 3, 4, 5]
            )
            metrics = survival_metrics(train, test, risk, surv)
            records.append({
                "feature_set": feature_name,
                "model": model_name,
                "n_features": len(used),
                **metrics,
                "n": len(test),
                "events": int(test["event"].sum()),
                "status": "ok",
            })
        except Exception as exc:
            records.append({
                "feature_set": feature_name,
                "model": model_name,
                "n_features": len(features),
                "auc_2yr": np.nan,
                "auc_5yr": np.nan,
                "c_index": np.nan,
                "integrated_brier_score_1to5": np.nan,
                "n": len(test),
                "events": int(test["event"].sum()),
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(records)


def survival_model_selection(frame, features, selected_feature):
    """Compare five algorithms only after the feature set is fixed."""
    train = administrative_training(
        frame, SURVIVAL_SELECTION_TRAIN_CUTOFF
    )
    test = administrative_test_cohort(
        frame,
        SURVIVAL_SELECTION_ORIGIN_START,
        SURVIVAL_SELECTION_ORIGIN_END,
        SURVIVAL_SELECTION_OUTCOME_CUTOFF,
    )
    records = []
    for model_name in SURVIVAL_MODELS:
        print(f"  survival model selection: {model_name}", flush=True)
        try:
            fitted, _, _, _, X_test_model, used, _ = fit_survival(
                train, test, features, model_name, selection=True
            )
            risk, surv = survival_predictions(
                fitted, model_name, X_test_model, [1, 2, 3, 4, 5]
            )
            records.append({
                "feature_set": selected_feature,
                "model": model_name,
                "n_features": len(used),
                **survival_metrics(train, test, risk, surv),
                "n": len(test),
                "events": int(test["event"].sum()),
                "status": "ok",
            })
        except Exception as exc:
            records.append({
                "feature_set": selected_feature,
                "model": model_name,
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(records)


def choose_survival_feature(selection):
    valid = selection[selection["status"].eq("ok")].copy()
    valid["mean_horizon_auc"] = valid[["auc_2yr", "auc_5yr"]].mean(axis=1)
    summary = (
        valid.groupby("feature_set", as_index=False)
        .agg(
            mean_selection_auc=("mean_horizon_auc", "mean"),
            minimum_selection_auc=("mean_horizon_auc", "min"),
            n_features=("n_features", "max"),
            successful_models=("model", "nunique"),
        )
    )
    best = float(summary["mean_selection_auc"].max())
    eligible = summary[
        summary["mean_selection_auc"].ge(best - FEATURE_SELECTION_TOLERANCE)
    ].copy()
    chosen = eligible.sort_values(
        ["n_features", "mean_selection_auc"], ascending=[True, False]
    ).iloc[0]["feature_set"]
    summary["within_selection_tolerance"] = summary[
        "mean_selection_auc"
    ].ge(best - FEATURE_SELECTION_TOLERANCE)
    summary["selected"] = summary["feature_set"].eq(chosen)
    return str(chosen), summary


def choose_survival_model(selection, feature_name):
    data = selection[
        selection["feature_set"].eq(feature_name)
        & selection["status"].eq("ok")
    ].copy()
    data["mean_horizon_auc"] = data[["auc_2yr", "auc_5yr"]].mean(axis=1)
    return str(data.sort_values(
        ["mean_horizon_auc", "c_index", "model"],
        ascending=[False, False, True],
    ).iloc[0]["model"])


def evaluate_one_survival_configuration(
    train, test, features, feature_name, model_name
):
    fitted, _, _, _, X_test_model, used, _ = fit_survival(
        train, test, features, model_name, selection=False
    )
    risk, surv = survival_predictions(
        fitted, model_name, X_test_model, [1, 2, 3, 4, 5]
    )
    return {
        "feature_set": feature_name,
        "model": model_name,
        "n_features": len(used),
        **survival_metrics(train, test, risk, surv),
        "n": len(test),
        "events": int(test["event"].sum()),
        "status": "ok",
    }


def survival_evaluation(frame, feature_sets, selection, selected_feature):
    train = administrative_training(
        frame, SURVIVAL_EVALUATION_TRAIN_CUTOFF
    )
    # Origins 2019-2020 provide at least one cohort with >5 years of support,
    # allowing time-dependent AUC at both 2 and 5 years.
    test = administrative_test_cohort(
        frame, 2019, 2020, SURVIVAL_EVALUATION_OUTCOME_CUTOFF
    )
    vertical = []
    for feature_name, features in feature_sets.items():
        model_name = "cox_ph"
        print(f"  locked survival vertical: {feature_name}", flush=True)
        try:
            vertical.append(evaluate_one_survival_configuration(
                train, test, features, feature_name, model_name
            ))
        except Exception as exc:
            vertical.append({
                "feature_set": feature_name,
                "model": model_name,
                "status": f"failed: {type(exc).__name__}: {exc}",
            })

    horizontal = []
    for model_name in SURVIVAL_MODELS:
        print(f"  locked survival horizontal: {model_name}", flush=True)
        try:
            horizontal.append(evaluate_one_survival_configuration(
                train,
                test,
                feature_sets[selected_feature],
                selected_feature,
                model_name,
            ))
        except Exception as exc:
            horizontal.append({
                "feature_set": selected_feature,
                "model": model_name,
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(vertical), pd.DataFrame(horizontal)


def add_vertical_prediction_tables(
    vertical_df, frame, feature_sets, selection_metrics, oof_predictions
):
    tables = {}
    for horizon in HORIZONS:
        target = f"target_within_{horizon}yr"
        data = frame.dropna(subset=[target]).copy()
        data["_target"] = data[target].astype(int)
        data = add_target_end_season(data, horizon)
        windows = temporal_windows(data)
        train = windows["evaluation_train"]
        test = windows["evaluation"]
        for feature_name in ("age_only", "minutes_only"):
            model_name = choose_model_for_feature(
                selection_metrics[
                    selection_metrics["horizon_years"].eq(horizon)
                ],
                feature_name,
            )
            oof = oof_predictions[(horizon, feature_name, model_name)]
            calibrator = fit_platt(oof["target"], oof["raw_probability"])
            _, raw, _, _ = fit_classification(
                train, test, feature_sets[feature_name], model_name
            )
            calibrated = apply_platt(calibrator, raw)
            tables[(horizon, feature_name, model_name)] = pd.DataFrame({
                "player_id": test["player_id"].to_numpy(),
                "season_start": test["season_start"].to_numpy(),
                "target": test["_target"].to_numpy(),
                "baseline_probability": calibrated,
            })
    vertical_df.attrs["prediction_tables"] = tables
    return vertical_df


def validate_expected_leagues(features):
    expected_leagues = {
        "GB1", "GB2", "ES1", "ES2", "L1", "L2", "IT1", "IT2",
        "FR1", "FR2", "NL1", "PO1", "BE1",
    }
    observed_leagues = set(
        features["player_domestic_comp"].dropna().astype(str).unique()
    )
    missing_leagues = sorted(expected_leagues - observed_leagues)
    if missing_leagues:
        raise RuntimeError(
            "The feature table is not aligned to the 13-league weighted data; "
            f"missing {missing_leagues}. Re-run scripts 02, 03 and 08 first."
        )
    return observed_leagues, expected_leagues


def run_classification_task(task, config, force_selection=False):
    data_path = OUT_DIR / CLASSIFICATION_TASK_FILES[task]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}; re-run 03_feature_engineering.py"
        )
    classification = pd.read_parquet(data_path)
    if "prediction_task" not in classification:
        raise RuntimeError(f"{data_path.name} has no prediction_task audit column")
    observed_tasks = set(classification["prediction_task"].dropna().unique())
    if observed_tasks != {task}:
        raise RuntimeError(
            f"Unexpected task values in {data_path.name}: {sorted(observed_tasks)}"
        )

    classification_targets = [f"target_within_{h}yr" for h in HORIZONS]
    class_features = base_feature_columns(
        classification,
        target_cols=classification_targets,
        extra_exclude={
            "seasons_ahead", "target_end_season", "_target", "prediction_task"
        },
    )
    class_sets = define_feature_sets(class_features)
    output_dir = MODEL_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = classification_cache_signature(data_path, config, class_sets)

    cached_selection = None if force_selection else (
        load_classification_selection_cache(output_dir, signature)
    )
    if cached_selection is None:
        print(f"Running {task} classification temporal selection...")
        class_selection, class_oof = classification_selection(
            classification, class_sets, config
        )
        save_classification_selection_cache(
            class_selection, class_oof, output_dir, signature
        )
    else:
        print(f"Loading validated {task} selection cache...")
        class_selection, class_oof = cached_selection
    selected_class_features, class_feature_summary = choose_feature_set(
        class_selection
    )
    print(
        f"Selected {task} classification feature set: "
        f"{selected_class_features}"
    )

    (
        class_vertical,
        class_horizontal,
        class_predictions,
        selected_class_models,
    ) = classification_evaluation(
        classification,
        class_sets,
        class_selection,
        class_oof,
        selected_class_features,
    )
    class_vertical = add_vertical_prediction_tables(
        class_vertical,
        classification,
        class_sets,
        class_selection,
        class_oof,
    )
    class_bootstrap = cluster_bootstrap_classification(
        class_vertical,
        class_predictions,
        class_selection,
        selected_class_features,
    )
    outputs = {
        "final_classification_selection_folds.csv": class_selection,
        "final_classification_feature_selection.csv": class_feature_summary,
        "final_classification_vertical_comparison.csv": class_vertical,
        "final_classification_model_comparison.csv": class_horizontal,
        "final_classification_evaluation_predictions.csv": class_predictions,
        "final_classification_cluster_bootstrap.csv": class_bootstrap,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
        # Preserve the historical root-level interface for the confirmation task.
        if task == "confirmation":
            frame.to_csv(MODEL_DIR / filename, index=False)
    return {
        "task": task,
        "classification": classification,
        "features": class_features,
        "feature_sets": class_sets,
        "selected_feature_set": selected_class_features,
        "selected_models": selected_class_models,
        "cache_signature": signature,
        "output_dir": output_dir,
    }


def run_survival_experiment(survival, class_features):
    # The landmark file carries extra label-audit fields; intersecting with the
    # prospective classification vocabulary keeps those fields out.
    survival_features = [
        column for column in class_features if column in survival.columns
    ]
    survival_sets = define_feature_sets(survival_features)

    print("Running survival feature-set selection...")
    survival_selection_df = survival_selection(survival, survival_sets)
    selected_survival_features, survival_feature_summary = (
        choose_survival_feature(survival_selection_df)
    )
    print(f"Selected survival feature set: {selected_survival_features}")
    survival_model_selection_df = survival_model_selection(
        survival,
        survival_sets[selected_survival_features],
        selected_survival_features,
    )
    survival_vertical, survival_horizontal = survival_evaluation(
        survival,
        survival_sets,
        survival_selection_df,
        selected_survival_features,
    )

    outputs = {
        "final_survival_feature_selection_models.csv": survival_selection_df,
        "final_survival_feature_selection.csv": survival_feature_summary,
        "final_survival_model_selection.csv": survival_model_selection_df,
        "final_survival_vertical_comparison.csv": survival_vertical,
        "final_survival_model_comparison.csv": survival_horizontal,
    }
    for filename, frame in outputs.items():
        frame.to_csv(MODEL_DIR / filename, index=False)
    return {
        "feature_sets": survival_sets,
        "selected_feature_set": selected_survival_features,
        "selected_model": choose_survival_model(
            survival_model_selection_df, selected_survival_features
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("both", *CLASSIFICATION_TASK_FILES),
        default="both",
        help="Classification task(s) to run.",
    )
    parser.add_argument(
        "--force-selection",
        action="store_true",
        help="Ignore a valid task-specific temporal-selection cache.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_uncertainty_config()
    features = pd.read_parquet(OUT_DIR / "player_season_features.parquet")
    survival = pd.read_parquet(OUT_DIR / "survival_landmark.parquet")
    observed_leagues, expected_leagues = validate_expected_leagues(features)

    tasks = tuple(CLASSIFICATION_TASK_FILES) if args.task == "both" else (args.task,)
    classification_results = {
        task: run_classification_task(task, config, args.force_selection)
        for task in tasks
    }
    reference_task = "confirmation" if "confirmation" in classification_results else tasks[0]
    reference = classification_results[reference_task]
    survival_result = run_survival_experiment(survival, reference["features"])

    task_manifest = {}
    for task, result in classification_results.items():
        classification = result["classification"]
        task_manifest[task] = {
            "dataset": CLASSIFICATION_TASK_FILES[task],
            "rows": int(len(classification)),
            "complete_career_peak_rows": int(
                classification["complete_career_peak_cohort"].eq(1).sum()
            ),
            "selected_feature_set": result["selected_feature_set"],
            "selected_models": {
                str(key): value for key, value in result["selected_models"].items()
            },
            "selection_cache_signature": result["cache_signature"],
        }

    manifest = {
        "operational_ground_truth": (
            "adjusted minutes below 65% of the positive prior peak for two "
            "adjacent seasons; event at second low season"
        ),
        "modeling_leagues": sorted(observed_leagues & expected_leagues),
        "player_season_rows": int(len(features)),
        "survival_landmark_rows": int(len(survival)),
        "primary_cohort": "all eligible at-risk player-seasons",
        "complete_career_peak_cohort_role": "sensitivity analysis only",
        "injury_data_used": False,
        "injury_exclusion_reason": (
            "legacy injury CSV was a synthetic test fixture; reconstructed "
            "public sources were not sufficiently homogeneous for the locked panel"
        ),
        "admissible_classification_feature_count": int(
            len(reference["feature_sets"]["full_model"])
        ),
        "horizons": list(HORIZONS),
        "classification_models": list(CLASSIFICATION_MODELS),
        "survival_models": list(SURVIVAL_MODELS),
        "classification_tasks": task_manifest,
        "feature_sets": reference["feature_sets"],
        "selected_classification_feature_set": reference["selected_feature_set"],
        "selected_classification_models": {
            str(key): value for key, value in reference["selected_models"].items()
        },
        "selected_survival_feature_set": survival_result["selected_feature_set"],
        "selected_survival_model": survival_result["selected_model"],
        "imputation": "training-fold median only",
        "classification_selection_target_years": "2018-2021",
        "classification_locked_evaluation_target_years": "2022-2023",
        "survival_selection_origins": "2013-2014; outcomes censored at 2019",
        "survival_locked_evaluation_origins": "2019-2020; outcomes censored at 2025",
    }
    with open(
        MODEL_DIR / "final_experiment_manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    for task, result in classification_results.items():
        per_task = dict(manifest)
        per_task["classification_task"] = task
        per_task["classification_rows"] = task_manifest[task]["rows"]
        per_task["complete_career_peak_rows_classification"] = task_manifest[
            task
        ]["complete_career_peak_rows"]
        per_task["selected_classification_feature_set"] = task_manifest[task][
            "selected_feature_set"
        ]
        per_task["selected_classification_models"] = task_manifest[task][
            "selected_models"
        ]
        (result["output_dir"] / "final_experiment_manifest.json").write_text(
            json.dumps(per_task, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
