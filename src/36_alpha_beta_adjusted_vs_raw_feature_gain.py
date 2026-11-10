"""Compare adjusted and raw trajectories under identical alpha/beta outcomes.

For every alpha/beta cell, this script keeps that cell's reconstructed
adjusted-minute drop-off target fixed and fits a raw-minute trajectory model on
the exact same temporal rows. It then joins those results to script 35's
scenario-adjusted trajectory results. This is the valid answer to: which
weighting cell gives the largest adjusted-vs-raw AUC difference?
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
# data/ 默认与代码目录同级；不存在时回退到上一级目录。
DATA_ROOT = BASE_DIR / "data" if (BASE_DIR / "data").is_dir() else BASE_DIR.parent / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
SENSITIVITY_SCRIPT = SCRIPT_DIR / "35_outcome_definition_and_weight_sensitivity.py"


def load_sensitivity_module():
    spec = importlib.util.spec_from_file_location("weight_sensitivity", SENSITIVITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load script 35 helpers.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_feature_columns(module) -> list[str]:
    return module.COMMON_FEATURES + [
        f"raw_{column}" for column in module.TRAJECTORY_FEATURES
    ]


def attach_raw_trajectory(base: pd.DataFrame, module) -> pd.DataFrame:
    raw = base.copy()
    raw["scenario_minutes"] = raw["total_minutes"]
    raw = module.add_trajectory_features(raw)
    rename = {column: f"raw_{column}" for column in module.TRAJECTORY_FEATURES}
    raw_values = raw[list(rename)].rename(columns=rename)
    return base.join(raw_values)


def evaluate_raw_features(frame: pd.DataFrame, scenario: dict, module, config: dict):
    records = []
    features = raw_feature_columns(module)
    for horizon in module.HORIZONS:
        target_column = f"target_within_{horizon}yr"
        data = frame.dropna(subset=[target_column]).copy()
        data["_target"] = data[target_column].astype(int)
        data = module.add_target_end_season(data, horizon)
        train = data[
            data["target_end_season"].le(config["selection_target_year_end"])
        ].copy()
        test = data[
            data["target_end_season"].between(
                config["evaluation_target_year_start"],
                config["evaluation_target_year_end"],
            )
        ].copy()
        X_train, X_test, _, used, _ = module.clean_feature_matrix(
            train, test, test, features
        )
        model = module.xgboost_classifier()
        model.fit(X_train, train["_target"].astype(int))
        probability = module.enforce_probability_bounds(
            model.predict_proba(X_test)[:, 1]
        )
        records.append({
            **scenario,
            "horizon_years": horizon,
            "feature_set": "matched_age_plus_raw_minutes_trajectory",
            "model": "xgboost_fixed_final_hyperparameters",
            "n_features_used": len(used),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_event_rate": float(test["_target"].mean()),
            "auc": float(roc_auc_score(test["_target"], probability)),
            "average_precision": float(
                average_precision_score(test["_target"], probability)
            ),
        })
    return records


def main() -> None:
    module = load_sensitivity_module()
    config = module.load_uncertainty_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(
        PROCESSED_DIR / "player_season_features.parquet",
        columns=module.STABLE_COLUMNS,
    ).sort_values(["player_id", "season_start"]).reset_index(drop=True)
    base = attach_raw_trajectory(base, module)

    canonical_frame = module.scenario_frame(
        base, base["final_weighted_minutes"], module.CANONICAL_RATIO
    )
    canonical_event = canonical_frame["scenario_event"].astype(int)
    if not canonical_event.equals(base["dropoff_event"].astype(int)):
        raise RuntimeError("Canonical adjusted labels no longer reproduce official labels.")

    match_columns = [
        "player_id", "season_start", "minutes_played", "match_quality_elo",
        "top5_reference_elo", "opponent_expected_score",
    ]
    matches = pd.read_parquet(
        PROCESSED_DIR / "match_level_weighted_minutes.parquet",
        columns=match_columns,
    )
    match_config = json.loads(
        (BASE_DIR / "config" / "match_quality_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    match_keys = matches[["player_id", "season_start"]]
    minutes_played = matches["minutes_played"].to_numpy(dtype=float)
    quality_signal = np.tanh(
        (matches["match_quality_elo"].to_numpy(dtype=float)
         - matches["top5_reference_elo"].to_numpy(dtype=float))
        / float(match_config["quality_elo_scale"])
    )
    opponent_signal = 2.0 * matches["opponent_expected_score"].to_numpy(dtype=float) - 1.0
    bounds = tuple(match_config["final_multiplier_bounds"])

    raw_records = []
    for alpha in module.ALPHAS:
        for beta in module.BETAS:
            print(f"Raw-feature evaluation alpha={alpha:.2f}, beta={beta:.2f}...", flush=True)
            if alpha == module.CANONICAL_ALPHA and beta == module.CANONICAL_BETA:
                frame = canonical_frame
            else:
                aggregate = module.aggregate_adjusted_minutes(
                    match_keys,
                    minutes_played,
                    quality_signal,
                    opponent_signal,
                    alpha,
                    beta,
                    bounds,
                )
                scenario_minutes = module.alignment_series(base, aggregate)
                frame = module.scenario_frame(
                    base, scenario_minutes, module.CANONICAL_RATIO
                )
            diagnostics = module.label_diagnostics(frame, canonical_event)
            raw_records.extend(evaluate_raw_features(
                frame,
                {
                    "scenario_family": "alpha_beta_raw_features",
                    "scenario_id": f"alpha_{alpha:.2f}_beta_{beta:.2f}",
                    "peak_ratio": module.CANONICAL_RATIO,
                    "alpha": alpha,
                    "beta": beta,
                    **diagnostics,
                },
                module,
                config,
            ))

    raw_metrics = pd.DataFrame(raw_records)
    raw_path = OUTPUT_DIR / "alpha_beta_raw_feature_auc_sensitivity_metrics.csv"
    raw_metrics.to_csv(raw_path, index=False)

    adjusted_path = OUTPUT_DIR / "alpha_beta_auc_sensitivity_metrics.csv"
    if not adjusted_path.exists():
        raise RuntimeError("Run script 35 before script 36.")
    adjusted = pd.read_csv(adjusted_path)
    adjusted = adjusted[[
        "alpha", "beta", "horizon_years", "auc", "average_precision",
        "n_train", "n_test", "test_event_rate", "first_confirmed_events",
    ]].rename(columns={
        "auc": "adjusted_trajectory_auc",
        "average_precision": "adjusted_trajectory_ap",
        "n_train": "adjusted_n_train",
        "n_test": "adjusted_n_test",
        "test_event_rate": "adjusted_test_event_rate",
        "first_confirmed_events": "adjusted_events",
    })
    raw = raw_metrics[[
        "alpha", "beta", "horizon_years", "auc", "average_precision",
        "n_train", "n_test", "test_event_rate", "first_confirmed_events",
    ]].rename(columns={
        "auc": "raw_trajectory_auc",
        "average_precision": "raw_trajectory_ap",
        "n_train": "raw_n_train",
        "n_test": "raw_n_test",
        "test_event_rate": "raw_test_event_rate",
        "first_confirmed_events": "raw_events",
    })
    gain = adjusted.merge(raw, on=["alpha", "beta", "horizon_years"], validate="one_to_one")
    same_targets = (
        gain["adjusted_n_train"].eq(gain["raw_n_train"])
        & gain["adjusted_n_test"].eq(gain["raw_n_test"])
        & np.isclose(gain["adjusted_test_event_rate"], gain["raw_test_event_rate"])
        & gain["adjusted_events"].eq(gain["raw_events"])
    )
    if not same_targets.all():
        raise RuntimeError("Raw and adjusted feature models do not share each scenario target.")
    gain["auc_gain_adjusted_minus_raw"] = (
        gain["adjusted_trajectory_auc"] - gain["raw_trajectory_auc"]
    )
    gain["ap_gain_adjusted_minus_raw"] = (
        gain["adjusted_trajectory_ap"] - gain["raw_trajectory_ap"]
    )
    gain_path = OUTPUT_DIR / "alpha_beta_adjusted_vs_raw_feature_gain.csv"
    gain.to_csv(gain_path, index=False)

    extrema = []
    for horizon, group in gain.groupby("horizon_years"):
        maximum = group.loc[group["auc_gain_adjusted_minus_raw"].idxmax()]
        minimum = group.loc[group["auc_gain_adjusted_minus_raw"].idxmin()]
        extrema.extend([
            {
                "horizon_years": horizon,
                "extreme": "maximum_adjusted_minus_raw_auc_gain",
                "alpha": maximum["alpha"],
                "beta": maximum["beta"],
                "auc_gain_adjusted_minus_raw": maximum["auc_gain_adjusted_minus_raw"],
                "adjusted_trajectory_auc": maximum["adjusted_trajectory_auc"],
                "raw_trajectory_auc": maximum["raw_trajectory_auc"],
            },
            {
                "horizon_years": horizon,
                "extreme": "minimum_adjusted_minus_raw_auc_gain",
                "alpha": minimum["alpha"],
                "beta": minimum["beta"],
                "auc_gain_adjusted_minus_raw": minimum["auc_gain_adjusted_minus_raw"],
                "adjusted_trajectory_auc": minimum["adjusted_trajectory_auc"],
                "raw_trajectory_auc": minimum["raw_trajectory_auc"],
            },
        ])
    extrema = pd.DataFrame(extrema)
    extrema_path = OUTPUT_DIR / "alpha_beta_adjusted_vs_raw_feature_gain_extrema.csv"
    extrema.to_csv(extrema_path, index=False)

    by_alpha = gain.groupby(["horizon_years", "alpha"], as_index=False).agg(
        mean_auc_gain=("auc_gain_adjusted_minus_raw", "mean"),
        min_auc_gain=("auc_gain_adjusted_minus_raw", "min"),
        max_auc_gain=("auc_gain_adjusted_minus_raw", "max"),
    )
    by_alpha_path = OUTPUT_DIR / "alpha_beta_adjusted_vs_raw_feature_gain_by_alpha.csv"
    by_alpha.to_csv(by_alpha_path, index=False)

    report_lines = [
        "# Alpha/Beta: Adjusted vs Raw Feature Gain",
        "",
        "Each row fixes the same alpha/beta-specific adjusted-minute outcome, "
        "landmarks and temporal split. Only the trajectory features differ: raw "
        "minutes versus that cell's adjusted minutes.",
        "",
        "## Largest AUC gaps",
        "",
        "| Horizon | Extreme | Alpha | Beta | Adjusted AUC | Raw AUC | Adjusted - raw |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in extrema.itertuples(index=False):
        report_lines.append(
            f"| {row.horizon_years}y | {row.extreme} | {row.alpha:.2f} | "
            f"{row.beta:.2f} | {row.adjusted_trajectory_auc:.3f} | "
            f"{row.raw_trajectory_auc:.3f} | {row.auc_gain_adjusted_minus_raw:+.3f} |"
        )
    report_lines.extend([
        "",
        "These grid results are sensitivity evidence, not a valid basis for "
        "choosing alpha/beta by the largest observed AUC difference.",
    ])
    report_path = OUTPUT_DIR / "alpha_beta_adjusted_vs_raw_feature_gain_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    for path in [raw_path, gain_path, extrema_path, by_alpha_path, report_path]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
