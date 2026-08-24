"""Create the final downstream interpretation package.

The upstream protocol is frozen. This script refits the already selected
XGBoost specification on the existing evaluation-training window solely to
extract TreeSHAP contributions and create publication-ready figures.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_COLORS = {2: "#167D9A", 5: "#D05A3A"}
POSITIVE_COLOR = "#C84B31"
NEGATIVE_COLOR = "#26738E"
NEUTRAL_COLOR = "#5F6670"

CASE_ORDER = [
    "high_confidence_true_positive",
    "high_confidence_true_negative",
    "high_confidence_false_positive",
    "high_confidence_false_negative",
]
CASE_LABELS = {
    "high_confidence_true_positive": "True positive",
    "high_confidence_true_negative": "True negative",
    "high_confidence_false_positive": "False positive",
    "high_confidence_false_negative": "False negative",
}

FEATURE_LABELS = {
    "age": "Age",
    "height": "Height",
    "career_seasons": "Career seasons",
    "games_played": "Games played",
    "career_games_played": "Career games played",
    "final_weighted_minutes": "Current adjusted minutes",
    "final_weighted_minutes_yoy": "Year-on-year minutes change",
    "final_weighted_minutes_avg2yr": "Prior two-year minutes average",
    "final_weighted_minutes_lag1": "Adjusted minutes one year ago",
    "final_weighted_minutes_lag2": "Adjusted minutes two years ago",
    "career_final_weighted_minutes": "Career adjusted minutes",
    "career_adjusted_minutes_per_season": "Career minutes per season",
    "previous_season_observed": "Previous season observed",
    "club_elo": "Club Elo",
    "club_rank_percentile": "Club rank percentile",
    "avg_match_quality_elo": "Average match-quality Elo",
    "avg_opponent_elo_gap": "Average opponent Elo gap",
    "injury_days_2yr": "Injury days over two years",
    "injury_data_available": "Injury data available",
}


def display_feature(name):
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def load_final_module():
    path = Path(__file__).with_name("29_final_experiment.py")
    spec = importlib.util.spec_from_file_location("final_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_final_xgboost(module, classification, features, horizon):
    target_col = f"target_within_{horizon}yr"
    data = classification.dropna(subset=[target_col]).copy()
    data["_target"] = data[target_col].astype(int)
    data = module.add_target_end_season(data, horizon)
    windows = module.temporal_windows(data)
    train = windows["evaluation_train"]
    test = windows["evaluation"]
    X_train, X_test, _, used, medians = module.clean_feature_matrix(
        train, test, test, features
    )
    model = module.classification_model("xgboost")
    model.fit(X_train, train["_target"].astype(int))
    return model, X_train, X_test, train, test, used, medians


def tree_shap_tables(model, X_test, test, used, horizon, cases):
    matrix = xgb.DMatrix(X_test, feature_names=used)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)
    margins = model.get_booster().predict(matrix, output_margin=True)
    if not np.allclose(contributions.sum(axis=1), margins, atol=1e-5):
        raise RuntimeError(f"TreeSHAP additivity failed for {horizon}-year model")

    shap_values = pd.DataFrame(
        contributions[:, :-1], columns=used, index=X_test.index
    )
    global_rows = []
    for feature in used:
        global_rows.append({
            "horizon_years": horizon,
            "feature": feature,
            "mean_absolute_shap_log_odds": float(
                shap_values[feature].abs().mean()
            ),
            "mean_signed_shap_log_odds": float(shap_values[feature].mean()),
            "feature_value_shap_spearman": float(
                X_test[feature].corr(shap_values[feature], method="spearman")
            ),
        })

    lookup = test[["player_id", "season_start", "_target"]].copy()
    lookup["test_index"] = lookup.index
    selected_cases = cases[cases["horizon_years"].eq(horizon)].merge(
        lookup,
        on=["player_id", "season_start"],
        how="left",
        validate="many_to_one",
    )
    if selected_cases["test_index"].isna().any():
        raise RuntimeError("Some held-out cases could not be aligned to X_test")

    local_rows = []
    for _, case in selected_cases.iterrows():
        index = int(case["test_index"])
        for feature in used:
            local_rows.append({
                "horizon_years": horizon,
                "case_type": case["case_type"],
                "player_id": case["player_id"],
                "name": case["name"],
                "season_start": case["season_start"],
                "target": case["target"],
                "calibrated_probability": case["calibrated_probability"],
                "feature": feature,
                "feature_value": float(X_test.loc[index, feature]),
                "shap_log_odds_contribution": float(
                    shap_values.loc[index, feature]
                ),
                "absolute_contribution": float(
                    abs(shap_values.loc[index, feature])
                ),
            })
    return pd.DataFrame(global_rows), pd.DataFrame(local_rows)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D8DCE1", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_permutation_importance(frame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), constrained_layout=True)
    for ax, horizon in zip(axes, (2, 5)):
        part = (
            frame[frame["horizon_years"].eq(horizon)]
            .nlargest(12, "permutation_auc_decrease_mean")
            .sort_values("permutation_auc_decrease_mean")
        )
        ax.barh(
            part["feature"].map(display_feature),
            part["permutation_auc_decrease_mean"],
            xerr=part["permutation_auc_decrease_std"],
            color=HORIZON_COLORS[horizon],
            alpha=0.9,
            error_kw={"ecolor": "#3B4046", "capsize": 2, "linewidth": 0.8},
        )
        ax.set_title(f"{horizon}-year drop-off", fontsize=14, weight="bold")
        ax.set_xlabel("Decrease in held-out AUC after permutation")
        style_axes(ax)
    fig.suptitle(
        "Model-Agnostic Feature Importance",
        fontsize=18,
        weight="bold",
    )
    fig.savefig(
        FIGURE_DIR / "final_permutation_importance_2yr_5yr.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_global_shap(frame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), constrained_layout=True)
    for ax, horizon in zip(axes, (2, 5)):
        part = (
            frame[frame["horizon_years"].eq(horizon)]
            .nlargest(12, "mean_absolute_shap_log_odds")
            .sort_values("mean_absolute_shap_log_odds")
        )
        ax.barh(
            part["feature"].map(display_feature),
            part["mean_absolute_shap_log_odds"],
            color=HORIZON_COLORS[horizon],
            alpha=0.9,
        )
        ax.set_title(f"{horizon}-year drop-off", fontsize=14, weight="bold")
        ax.set_xlabel("Mean absolute TreeSHAP contribution (log-odds)")
        style_axes(ax)
    fig.suptitle("Global XGBoost TreeSHAP Importance", fontsize=18, weight="bold")
    fig.savefig(
        FIGURE_DIR / "final_global_treeshap_2yr_5yr.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_local_cases(local):
    selected = (
        local.sort_values("absolute_contribution", ascending=False)
        .groupby(["horizon_years", "case_type"], as_index=False)
        .head(6)
    )
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.5), constrained_layout=True)
    for row, horizon in enumerate((2, 5)):
        for col, case_type in enumerate(CASE_ORDER):
            ax = axes[row, col]
            part = selected[
                selected["horizon_years"].eq(horizon)
                & selected["case_type"].eq(case_type)
            ].sort_values("shap_log_odds_contribution")
            if part.empty:
                ax.set_visible(False)
                continue
            colors = np.where(
                part["shap_log_odds_contribution"].ge(0),
                POSITIVE_COLOR,
                NEGATIVE_COLOR,
            )
            ax.barh(
                part["feature"].map(display_feature),
                part["shap_log_odds_contribution"],
                color=colors,
            )
            ax.axvline(0, color="#343A40", linewidth=0.8)
            first = part.iloc[0]
            ax.set_title(
                f"{CASE_LABELS[case_type]}\n{first['name']} | risk "
                f"{first['calibrated_probability']:.1%}",
                fontsize=11,
                weight="bold",
            )
            ax.set_xlabel("Contribution to model log-odds")
            ax.grid(axis="x", color="#D8DCE1", linewidth=0.6, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.text(
                    -0.35,
                    1.18,
                    f"{horizon}-year model",
                    transform=ax.transAxes,
                    fontsize=15,
                    weight="bold",
                )
    fig.suptitle(
        "Held-Out Player Cases: Features Raising and Lowering Predicted Risk",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Red raises model risk; blue lowers model risk. Risks shown are Platt-calibrated probabilities.",
        ha="center",
        fontsize=11,
        color=NEUTRAL_COLOR,
    )
    fig.savefig(
        FIGURE_DIR / "final_local_treeshap_player_cases.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    for horizon in (2, 5):
        fig, axes = plt.subplots(1, 4, figsize=(19, 5.8), constrained_layout=True)
        for ax, case_type in zip(axes, CASE_ORDER):
            part = selected[
                selected["horizon_years"].eq(horizon)
                & selected["case_type"].eq(case_type)
            ].sort_values("shap_log_odds_contribution")
            colors = np.where(
                part["shap_log_odds_contribution"].ge(0),
                POSITIVE_COLOR,
                NEGATIVE_COLOR,
            )
            ax.barh(
                part["feature"].map(display_feature),
                part["shap_log_odds_contribution"],
                color=colors,
            )
            ax.axvline(0, color="#343A40", linewidth=0.8)
            first = part.iloc[0]
            ax.set_title(
                f"{CASE_LABELS[case_type]}\n{first['name']} | risk "
                f"{first['calibrated_probability']:.1%}",
                fontsize=12,
                weight="bold",
            )
            ax.set_xlabel("Contribution to model log-odds")
            ax.grid(axis="x", color="#D8DCE1", linewidth=0.6, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"{horizon}-Year Held-Out Player Cases",
            fontsize=18,
            weight="bold",
        )
        fig.text(
            0.5,
            -0.02,
            "Red raises model risk; blue lowers model risk. Risks shown are Platt-calibrated probabilities.",
            ha="center",
            fontsize=11,
            color=NEUTRAL_COLOR,
        )
        fig.savefig(
            FIGURE_DIR / f"final_local_treeshap_cases_{horizon}yr.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_calibration(frame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    for ax, horizon in zip(axes, (2, 5)):
        part = frame[frame["horizon_years"].eq(horizon)]
        ax.plot([0, 1], [0, 1], "--", color="#7B8188", label="Perfect calibration")
        ax.plot(
            part["mean_predicted_risk"],
            part["observed_rate"],
            marker="o",
            linewidth=2.2,
            color=HORIZON_COLORS[horizon],
            label="XGBoost",
        )
        for _, point in part.iterrows():
            ax.annotate(
                str(int(point["risk_decile"])),
                (point["mean_predicted_risk"], point["observed_rate"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{horizon}-year risk", fontsize=14, weight="bold")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(color="#D8DCE1", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Locked-Test Calibration by Risk Decile", fontsize=18, weight="bold")
    fig.savefig(
        FIGURE_DIR / "final_calibration_deciles_2yr_5yr.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_league_auc(frame):
    order = (
        frame.groupby("league_name")["auc"].mean().sort_values().index.tolist()
    )
    y = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
    for offset, horizon in ((-width / 2, 2), (width / 2, 5)):
        values = (
            frame[frame["horizon_years"].eq(horizon)]
            .set_index("league_name")
            .reindex(order)["auc"]
        )
        ax.barh(
            y + offset,
            values,
            height=width,
            color=HORIZON_COLORS[horizon],
            label=f"{horizon}-year",
        )
    ax.set_yticks(y, order)
    ax.set_xlim(0.5, 0.9)
    ax.axvline(0.7, color="#7B8188", linestyle="--", linewidth=1)
    ax.set_xlabel("Held-out AUC")
    ax.set_title("Classification Robustness Across 13 Leagues", fontsize=18, weight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.savefig(
        FIGURE_DIR / "final_league_subgroup_auc.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_baseline_and_models(vertical, horizontal):
    baseline_sets = ["age_only", "minutes_only", "clinical_extended"]
    labels = {
        "age_only": "Age only",
        "minutes_only": "Minutes only",
        "clinical_extended": "Final features",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, horizon in zip(axes, (2, 5)):
        part = vertical[
            vertical["horizon_years"].eq(horizon)
            & vertical["feature_set"].isin(baseline_sets)
        ].set_index("feature_set").reindex(baseline_sets)
        ax.bar(
            [labels[x] for x in baseline_sets],
            part["auc"],
            color=["#747B83", "#A4AAB1", HORIZON_COLORS[horizon]],
        )
        for index, value in enumerate(part["auc"]):
            ax.text(index, value + 0.008, f"{value:.3f}", ha="center", fontsize=10)
        ax.set_ylim(0.5, 0.86)
        ax.set_ylabel("Held-out AUC")
        ax.set_title(f"{horizon}-year horizon", fontsize=14, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D8DCE1", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Final Model Against Naive Baselines", fontsize=18, weight="bold")
    fig.savefig(
        FIGURE_DIR / "final_classification_baseline_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    pivot = horizontal.pivot(index="model", columns="horizon_years", values="auc")
    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[order]
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.barh(y - 0.18, pivot[2], height=0.36, color=HORIZON_COLORS[2], label="2-year")
    ax.barh(y + 0.18, pivot[5], height=0.36, color=HORIZON_COLORS[5], label="5-year")
    model_labels = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "logistic": "Logistic Regression",
    }
    ax.set_yticks(y, [model_labels[name] for name in pivot.index])
    ax.set_xlim(0.5, 0.83)
    ax.set_xlabel("Held-out AUC")
    ax.set_title("Five Classification Models on Fixed Final Features", fontsize=17, weight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.savefig(
        FIGURE_DIR / "final_classification_five_model_auc.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    module = load_final_module()
    with open(OUTPUT_DIR / "final_experiment_manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected_set = manifest["selected_classification_feature_set"]
    selected_features = manifest["feature_sets"][selected_set]
    classification = pd.read_parquet(
        PROCESSED_DIR / "classification_format_yearly.parquet"
    )
    cases = pd.read_csv(OUTPUT_DIR / "final_heldout_player_cases.csv")
    representative_parts = []
    high_risk_types = {
        "high_confidence_true_positive",
        "high_confidence_false_positive",
    }
    for (horizon, case_type), group in cases.groupby(
        ["horizon_years", "case_type"], sort=False
    ):
        representative_parts.append(
            group.sort_values(
                "calibrated_probability",
                ascending=case_type not in high_risk_types,
            ).head(1)
        )
    representative_cases = pd.concat(representative_parts, ignore_index=True)

    global_parts = []
    local_parts = []
    for horizon in (2, 5):
        model, _, X_test, train, test, used, medians = fit_final_xgboost(
            module, classification, selected_features, horizon
        )
        global_frame, local_frame = tree_shap_tables(
            model,
            X_test,
            test,
            used,
            horizon,
            representative_cases,
        )
        global_parts.append(global_frame)
        local_parts.append(local_frame)
        joblib.dump(
            {
                "model": model,
                "features": used,
                "training_medians": medians,
                "horizon_years": horizon,
                "training_rows": len(train),
                "locked_test_rows": len(test),
            },
            OUTPUT_DIR / f"final_xgboost_{horizon}yr_interpretability_bundle.joblib",
        )

    global_shap = pd.concat(global_parts, ignore_index=True)
    local_shap = pd.concat(local_parts, ignore_index=True)
    global_shap.to_csv(
        OUTPUT_DIR / "final_xgboost_global_treeshap.csv", index=False
    )
    local_shap.to_csv(
        OUTPUT_DIR / "final_xgboost_local_case_treeshap.csv", index=False
    )

    permutation = pd.read_csv(
        OUTPUT_DIR / "final_xgboost_permutation_importance.csv"
    )
    calibration = pd.read_csv(
        OUTPUT_DIR / "final_classification_calibration_deciles.csv"
    )
    leagues = pd.read_csv(
        OUTPUT_DIR / "final_classification_league_subgroups.csv"
    )
    vertical = pd.read_csv(
        OUTPUT_DIR / "final_classification_vertical_comparison.csv"
    )
    horizontal = pd.read_csv(
        OUTPUT_DIR / "final_classification_model_comparison.csv"
    )

    plot_permutation_importance(permutation)
    plot_global_shap(global_shap)
    plot_local_cases(local_shap)
    plot_calibration(calibration)
    plot_league_auc(leagues)
    plot_baseline_and_models(vertical, horizontal)

    print("Top global TreeSHAP features:")
    print(
        global_shap.sort_values(
            ["horizon_years", "mean_absolute_shap_log_odds"],
            ascending=[True, False],
        ).groupby("horizon_years", group_keys=False).head(10).to_string(index=False)
    )
    print("\nRepresentative held-out cases:")
    print(
        representative_cases[[
            "horizon_years", "case_type", "name", "target",
            "calibrated_probability",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
