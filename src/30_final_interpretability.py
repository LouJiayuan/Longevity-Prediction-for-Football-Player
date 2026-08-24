"""Interpretability, calibration and 13-league subgroup audit.

This script is downstream of the locked experiment in 29_final_experiment.py.
It does not select features or models. It explains the pre-selected XGBoost
classification models on the held-out 2022-2023 target-end cohorts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

LEAGUES = {
    "GB1": ("Premier League", "England", 1),
    "GB2": ("Championship", "England", 2),
    "ES1": ("LaLiga", "Spain", 1),
    "ES2": ("LaLiga 2", "Spain", 2),
    "L1": ("Bundesliga", "Germany", 1),
    "L2": ("2. Bundesliga", "Germany", 2),
    "IT1": ("Serie A", "Italy", 1),
    "IT2": ("Serie B", "Italy", 2),
    "FR1": ("Ligue 1", "France", 1),
    "FR2": ("Ligue 2", "France", 2),
    "NL1": ("Eredivisie", "Netherlands", 1),
    "PO1": ("Primeira Liga", "Portugal", 1),
    "BE1": ("Belgian Pro League", "Belgium", 1),
}


def load_final_module():
    path = Path(__file__).with_name("29_final_experiment.py")
    spec = importlib.util.spec_from_file_location("final_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(group):
    y = group["target"].astype(int)
    p = group["calibrated_probability"].astype(float)
    result = {
        "n": len(group),
        "n_players": group["player_id"].nunique(),
        "events": int(y.sum()),
        "observed_rate": float(y.mean()),
        "mean_predicted_risk": float(p.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
        "brier": float(brier_score_loss(y, p)),
    }
    if y.nunique() == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    else:
        result["auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def add_context(predictions, features):
    context_cols = [
        "player_id",
        "season_start",
        "name",
        "position_group",
        "age",
        "final_weighted_minutes",
        "club_elo",
        "player_domestic_comp",
    ]
    out = predictions.merge(
        features[context_cols], on=["player_id", "season_start"], how="left",
        validate="many_to_one",
    )
    if out["player_domestic_comp"].isna().any():
        raise RuntimeError("League context failed to align for some predictions")
    out["league_name"] = out["player_domestic_comp"].map(
        lambda code: LEAGUES[code][0]
    )
    out["country"] = out["player_domestic_comp"].map(
        lambda code: LEAGUES[code][1]
    )
    out["league_tier"] = out["player_domestic_comp"].map(
        lambda code: LEAGUES[code][2]
    )
    return out


def subgroup_tables(predictions):
    league_rows = []
    tier_rows = []
    for horizon, horizon_frame in predictions.groupby("horizon_years"):
        for code, group in horizon_frame.groupby("player_domestic_comp"):
            league_rows.append({
                "horizon_years": int(horizon),
                "competition_code": code,
                "league_name": LEAGUES[code][0],
                "country": LEAGUES[code][1],
                "league_tier": LEAGUES[code][2],
                **metrics(group),
            })
        for tier, group in horizon_frame.groupby("league_tier"):
            tier_rows.append({
                "horizon_years": int(horizon),
                "league_tier": int(tier),
                **metrics(group),
            })
    return pd.DataFrame(league_rows), pd.DataFrame(tier_rows)


def calibration_table(predictions):
    rows = []
    for horizon, group in predictions.groupby("horizon_years"):
        bins = pd.qcut(
            group["calibrated_probability"], q=10, duplicates="drop"
        )
        for index, (_, part) in enumerate(group.groupby(bins, observed=True), 1):
            rows.append({
                "horizon_years": int(horizon),
                "risk_decile": index,
                "n": len(part),
                "mean_predicted_risk": part["calibrated_probability"].mean(),
                "observed_rate": part["target"].mean(),
                "calibration_gap": (
                    part["calibrated_probability"].mean()
                    - part["target"].mean()
                ),
            })
    return pd.DataFrame(rows)


def interesting_cases(predictions, n_each=5):
    rows = []
    for horizon, group in predictions.groupby("horizon_years"):
        group = group.copy()
        group["predicted_class_at_0_5"] = group[
            "calibrated_probability"
        ].ge(0.5).astype(int)
        definitions = {
            "high_confidence_true_positive": (
                group["target"].eq(1) & group["predicted_class_at_0_5"].eq(1),
                False,
            ),
            "high_confidence_true_negative": (
                group["target"].eq(0) & group["predicted_class_at_0_5"].eq(0),
                True,
            ),
            "high_confidence_false_positive": (
                group["target"].eq(0) & group["predicted_class_at_0_5"].eq(1),
                False,
            ),
            "high_confidence_false_negative": (
                group["target"].eq(1) & group["predicted_class_at_0_5"].eq(0),
                True,
            ),
        }
        for case_type, (mask, ascending) in definitions.items():
            selected = group[mask].sort_values(
                "calibrated_probability", ascending=ascending
            ).head(n_each).copy()
            selected["case_type"] = case_type
            rows.append(selected)
    columns = [
        "horizon_years", "case_type", "player_id", "name", "season_start",
        "position_group", "age", "league_name", "league_tier",
        "final_weighted_minutes", "club_elo", "target",
        "calibrated_probability", "predicted_class_at_0_5",
    ]
    return pd.concat(rows, ignore_index=True)[columns]


def final_feature_importance(module, classification, feature_columns):
    rows = []
    for horizon in (2, 5):
        target_col = f"target_within_{horizon}yr"
        data = classification.dropna(subset=[target_col]).copy()
        data["_target"] = data[target_col].astype(int)
        data = module.add_target_end_season(data, horizon)
        windows = module.temporal_windows(data)
        train = windows["evaluation_train"]
        test = windows["evaluation"]
        X_train, X_test, _, used, _ = module.clean_feature_matrix(
            train, test, test, feature_columns
        )
        model = module.classification_model("xgboost")
        model.fit(X_train, train["_target"].astype(int))
        result = permutation_importance(
            model,
            X_test,
            test["_target"].astype(int),
            scoring="roc_auc",
            n_repeats=5,
            random_state=42,
            n_jobs=4,
        )
        for feature, mean, std in zip(
            used, result.importances_mean, result.importances_std
        ):
            rows.append({
                "horizon_years": horizon,
                "feature": feature,
                "permutation_auc_decrease_mean": float(mean),
                "permutation_auc_decrease_std": float(std),
            })
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["horizon_years", "permutation_auc_decrease_mean"],
        ascending=[True, False],
    )


def main():
    module = load_final_module()
    with open(
        OUTPUT_DIR / "final_experiment_manifest.json", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    if manifest["selected_classification_models"] != {
        "2": "xgboost", "5": "xgboost"
    }:
        raise RuntimeError("Interpretability script expects final XGBoost models")

    predictions = pd.read_csv(
        OUTPUT_DIR / "final_classification_evaluation_predictions.csv"
    )
    predictions = predictions[
        predictions["model"].eq("xgboost")
    ].copy()
    features = pd.read_parquet(
        PROCESSED_DIR / "player_season_features.parquet"
    )
    classification = pd.read_parquet(
        PROCESSED_DIR / "classification_format_yearly.parquet"
    )
    contextual = add_context(predictions, features)

    league, tier = subgroup_tables(contextual)
    calibration = calibration_table(contextual)
    cases = interesting_cases(contextual)
    importance = final_feature_importance(
        module,
        classification,
        manifest["feature_sets"][manifest["selected_classification_feature_set"]],
    )

    contextual.to_csv(
        OUTPUT_DIR / "final_classification_predictions_with_context.csv",
        index=False,
    )
    league.to_csv(
        OUTPUT_DIR / "final_classification_league_subgroups.csv", index=False
    )
    tier.to_csv(
        OUTPUT_DIR / "final_classification_tier_robustness.csv", index=False
    )
    calibration.to_csv(
        OUTPUT_DIR / "final_classification_calibration_deciles.csv", index=False
    )
    cases.to_csv(OUTPUT_DIR / "final_heldout_player_cases.csv", index=False)
    importance.to_csv(
        OUTPUT_DIR / "final_xgboost_permutation_importance.csv", index=False
    )

    print(league.to_string(index=False))
    print("\nTier robustness:\n", tier.to_string(index=False))
    print("\nTop feature importance:\n", importance.groupby(
        "horizon_years", group_keys=False
    ).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
