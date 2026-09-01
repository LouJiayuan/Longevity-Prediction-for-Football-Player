#!/usr/bin/env python3
"""Compare verified public injury-history features with the final baselines.

This experiment replaces the invalid legacy ``clinical`` feature sets. It keeps
the 28 August temporal protocol and model families, but adds only injury proxy
features derived from the audited longitudinal alignment. No feature uses the
``history_zero_assumption`` because knowing that a player has any event later in
their career would leak future information into earlier landmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


KIMI_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = KIMI_DIR / "data" / "processed"
OUTPUT_DIR = KIMI_DIR / "outputs" / "verified_injury_feature_comparison"
INJURY_PATH = (
    KIMI_DIR.parent
    / "data"
    / "injuries"
    / "longitudinal_2000_2025"
    / "player_season_injury_features.parquet"
)
FINAL_SCRIPT = KIMI_DIR / "scripts" / "29_final_experiment.py"
CLASSIFICATION_TASK_FILES = {
    "confirmation": "classification_confirmation_yearly.parquet",
    "early_warning": "classification_early_warning_yearly.parquet",
}


def load_final_experiment_module():
    spec = importlib.util.spec_from_file_location(
        "final_experiment_for_verified_injury", FINAL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {FINAL_SCRIPT}")
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_injury_feature_columns() -> list[str]:
    return [
        "injury_days_current",
        "injury_absence_count_current",
        "injury_severe_count_current",
        "injury_major_count_current",
        "injury_games_missed_current",
        "injury_days_lag1",
        "injury_days_2yr",
        "injury_absence_count_2yr",
        "injury_career_days",
        "injury_source_profile_present",
    ]


def build_verified_injury_features(source: pd.DataFrame) -> pd.DataFrame:
    """Create prospective player-season injury proxies without future flags."""
    required = {
        "player_id",
        "season_start",
        "source_profile_present",
        "source_profile_dob_conflict",
        "injury_event_recorded_this_season",
        "injury_main_window",
        "injury_sensitivity_window",
        "validated_unique_injury_days_profile_zero_assumption",
        "validated_injury_start_count_profile_zero_assumption",
        "severe_injury_start_count_profile_zero_assumption",
        "major_injury_start_count_profile_zero_assumption",
        "reported_games_missed_at_start_profile_zero_assumption",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Verified injury source is missing: {sorted(missing)}")

    forbidden = [
        column for column in source.columns if "history_zero_assumption" in column
    ]
    if not forbidden:
        raise ValueError("Expected audited history-zero fields were not found")

    frame = source.sort_values(["player_id", "season_start"]).copy()
    accepted_profile = (
        frame["source_profile_present"].fillna(False).astype(bool)
        & ~frame["source_profile_dob_conflict"].fillna(False).astype(bool)
    )
    frame["injury_source_profile_present"] = accepted_profile.astype(int)

    rename = {
        "validated_unique_injury_days_profile_zero_assumption": "injury_days_current",
        "validated_injury_start_count_profile_zero_assumption": "injury_absence_count_current",
        "severe_injury_start_count_profile_zero_assumption": "injury_severe_count_current",
        "major_injury_start_count_profile_zero_assumption": "injury_major_count_current",
        "reported_games_missed_at_start_profile_zero_assumption": "injury_games_missed_current",
    }
    for source_column, output_column in rename.items():
        frame[output_column] = pd.to_numeric(
            frame[source_column], errors="coerce"
        )

    grouped = frame.groupby("player_id", sort=False, group_keys=False)
    frame["injury_days_lag1"] = grouped["injury_days_current"].shift(1)
    frame["injury_days_2yr"] = grouped["injury_days_current"].transform(
        lambda values: values.rolling(2, min_periods=1).sum()
    )
    frame["injury_absence_count_2yr"] = grouped[
        "injury_absence_count_current"
    ].transform(lambda values: values.rolling(2, min_periods=1).sum())
    frame["injury_career_days"] = grouped["injury_days_current"].transform(
        lambda values: values.cumsum()
    )
    feature_columns = verified_injury_feature_columns()
    output = frame[["player_id", "season_start", *feature_columns]].copy()
    if output.duplicated(["player_id", "season_start"]).any():
        raise ValueError("Verified injury features contain duplicate player-seasons")
    return output


def merge_verified_injury_features(
    frame: pd.DataFrame, injury_features: pd.DataFrame
) -> pd.DataFrame:
    original_rows = len(frame)
    merged = frame.merge(
        injury_features,
        on=["player_id", "season_start"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != original_rows:
        raise RuntimeError("Injury feature merge changed the modelling row count")
    return merged


def define_comparison_sets(final_module, frame: pd.DataFrame) -> dict[str, list[str]]:
    target_columns = [f"target_within_{horizon}yr" for horizon in (2, 5)]
    base_features = final_module.base_feature_columns(
        frame,
        target_cols=target_columns,
        extra_exclude={
            "seasons_ahead", "target_end_season", "_target", "prediction_task"
        },
    )
    base_sets = final_module.define_feature_sets(base_features)
    injury = [
        column for column in verified_injury_feature_columns() if column in frame
    ]

    def combine(*parts):
        return list(dict.fromkeys(column for part in parts for column in part))

    comparison = dict(base_sets)
    comparison.update({
        "injury_only": injury,
        "age_injury": combine(base_sets["age_only"], ["age_sq"], injury),
        "minutes_injury": combine(base_sets["minutes_only"], injury),
        "age_minutes_injury": combine(base_sets["age_plus_minutes"], injury),
        "age_minutes_elo_injury": combine(base_sets["age_minutes_elo"], injury),
        "full_model_injury": combine(base_sets["full_model"], injury),
    })
    if any(name in comparison for name in ("clinical", "clinical_extended")):
        raise RuntimeError("Legacy clinical feature-set names must not be restored")
    if any("history_zero_assumption" in column for columns in comparison.values() for column in columns):
        raise RuntimeError("Future-history zero assumptions entered the feature sets")
    return comparison


def cache_signature(
    data_path: Path,
    injury_path: Path,
    config: dict[str, object],
    feature_sets: dict[str, list[str]],
) -> str:
    payload = {
        "data_sha256": sha256_file(data_path),
        "injury_sha256": sha256_file(injury_path),
        "config": config,
        "feature_sets": feature_sets,
        "schema": 1,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_baseline_selection(final_module, task: str, data_path: Path, config, base_sets):
    baseline_dir = KIMI_DIR / "outputs" / task
    expected = final_module.classification_cache_signature(
        data_path, config, base_sets
    )
    cached = final_module.load_classification_selection_cache(
        baseline_dir, expected
    )
    if cached is None:
        raise RuntimeError(
            f"The validated non-injury selection cache for {task} is missing or stale. "
            "Run scripts/29_final_experiment.py first."
        )
    return cached


def run_classification_task(
    final_module,
    task: str,
    injury_features: pd.DataFrame,
    config: dict[str, object],
    force_selection: bool,
):
    data_path = DATA_DIR / CLASSIFICATION_TASK_FILES[task]
    frame = merge_verified_injury_features(
        pd.read_parquet(data_path), injury_features
    )
    comparison_sets = define_comparison_sets(final_module, frame)
    base_names = set(final_module.define_feature_sets(
        final_module.base_feature_columns(
            frame,
            target_cols=[f"target_within_{horizon}yr" for horizon in (2, 5)],
            extra_exclude={
                "seasons_ahead", "target_end_season", "_target", "prediction_task"
            },
        )
    ))
    base_sets = {name: comparison_sets[name] for name in base_names}
    injury_sets = {
        name: columns
        for name, columns in comparison_sets.items()
        if name not in base_names
    }
    baseline_metrics, baseline_oof = load_baseline_selection(
        final_module, task, data_path, config, base_sets
    )

    task_dir = OUTPUT_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(data_path, INJURY_PATH, config, injury_sets)
    cached = None if force_selection else final_module.load_classification_selection_cache(
        task_dir, signature
    )
    if cached is None:
        print(f"Running verified injury selection for {task}...", flush=True)
        injury_metrics, injury_oof = final_module.classification_selection(
            frame, injury_sets, config
        )
        final_module.save_classification_selection_cache(
            injury_metrics, injury_oof, task_dir, signature
        )
    else:
        print(f"Loading verified injury selection cache for {task}...", flush=True)
        injury_metrics, injury_oof = cached

    combined_metrics = pd.concat(
        [baseline_metrics, injury_metrics], ignore_index=True
    )
    combined_oof = dict(baseline_oof)
    combined_oof.update(injury_oof)
    selected_feature, feature_summary = final_module.choose_feature_set(
        combined_metrics
    )
    print(f"Selected {task} feature set: {selected_feature}", flush=True)
    vertical_path = task_dir / "verified_injury_classification_vertical_comparison.csv"
    horizontal_path = task_dir / "verified_injury_classification_model_comparison.csv"
    predictions_path = task_dir / "verified_injury_classification_predictions.parquet"
    evaluation_cached = (
        not force_selection
        and cached is not None
        and vertical_path.exists()
        and horizontal_path.exists()
        and predictions_path.exists()
    )
    if evaluation_cached:
        print(f"Loading verified injury locked evaluation for {task}...", flush=True)
        vertical = pd.read_csv(vertical_path)
        horizontal = pd.read_csv(horizontal_path)
        predictions = pd.read_parquet(predictions_path)
        selected_models = {
            horizon: final_module.choose_model_for_feature(
                combined_metrics[
                    combined_metrics["horizon_years"].eq(horizon)
                ],
                selected_feature,
            )
            for horizon in (2, 5)
        }
    else:
        vertical, horizontal, predictions, selected_models = (
            final_module.classification_evaluation(
                frame,
                comparison_sets,
                combined_metrics,
                combined_oof,
                selected_feature,
            )
        )
    combined_metrics.to_csv(
        task_dir / "verified_injury_classification_selection_folds.csv",
        index=False,
    )
    feature_summary.to_csv(
        task_dir / "verified_injury_classification_feature_selection.csv",
        index=False,
    )
    vertical.to_csv(vertical_path, index=False)
    horizontal.to_csv(horizontal_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    return {
        "task": task,
        "frame": frame,
        "feature_sets": comparison_sets,
        "selected_feature_set": selected_feature,
        "selected_models": selected_models,
        "vertical": vertical,
        "feature_summary": feature_summary,
        "cache_signature": signature,
    }


def run_survival(final_module, injury_features, comparison_sets):
    survival = merge_verified_injury_features(
        pd.read_parquet(DATA_DIR / "survival_landmark.parquet"),
        injury_features,
    )
    survival_sets = {
        name: [column for column in columns if column in survival.columns]
        for name, columns in comparison_sets.items()
    }
    print("Running verified injury survival feature selection...", flush=True)
    selection_models = final_module.survival_selection(survival, survival_sets)
    selected_feature, feature_summary = final_module.choose_survival_feature(
        selection_models
    )
    print(f"Selected survival feature set: {selected_feature}", flush=True)
    model_selection = final_module.survival_model_selection(
        survival,
        survival_sets[selected_feature],
        selected_feature,
    )
    vertical, horizontal = final_module.survival_evaluation(
        survival,
        survival_sets,
        selection_models,
        selected_feature,
    )
    selection_models.to_csv(
        OUTPUT_DIR / "verified_injury_survival_feature_selection_models.csv",
        index=False,
    )
    feature_summary.to_csv(
        OUTPUT_DIR / "verified_injury_survival_feature_selection.csv",
        index=False,
    )
    model_selection.to_csv(
        OUTPUT_DIR / "verified_injury_survival_model_selection.csv", index=False
    )
    vertical.to_csv(
        OUTPUT_DIR / "verified_injury_survival_vertical_comparison.csv", index=False
    )
    horizontal.to_csv(
        OUTPUT_DIR / "verified_injury_survival_model_comparison.csv", index=False
    )
    return {
        "selected_feature_set": selected_feature,
        "selected_model": final_module.choose_survival_model(
            model_selection, selected_feature
        ),
        "feature_summary": feature_summary,
        "vertical": vertical,
    }


def format_number(value, digits=3):
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def write_report(classification_results, survival_result, manifest):
    lines = [
        "# Verified Injury Feature Comparison",
        "",
        "This experiment replaces the invalid legacy `clinical` and "
        "`clinical_extended` sets. Injury variables are public "
        "availability-history proxies, not medical clinical data.",
        "",
        "## Classification locked evaluation",
        "",
        "| Task | Horizon | Feature set | Model | AUC | AP | Brier | N |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for task, result in classification_results.items():
        vertical = result["vertical"].sort_values(
            ["horizon_years", "auc"], ascending=[True, False]
        )
        for row in vertical.itertuples(index=False):
            lines.append(
                f"| {task.replace('_', ' ').title()} | {int(row.horizon_years)} | "
                f"{row.feature_set} | {row.selected_model_from_selection} | "
                f"{format_number(row.auc)} | {format_number(row.average_precision)} | "
                f"{format_number(row.brier)} | {int(row.n):,} |"
            )
    lines.extend([
        "",
        "## Survival locked evaluation",
        "",
        "| Feature set | Model | C-index | AUC 2y | AUC 5y | IBS |",
        "|---|---|---:|---:|---:|---:|",
    ])
    survival_vertical = survival_result["vertical"].copy()
    survival_vertical = survival_vertical.sort_values(
        "c_index", ascending=False, na_position="last"
    )
    for row in survival_vertical.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.model} | {format_number(row.c_index)} | "
            f"{format_number(row.auc_2yr)} | {format_number(row.auc_5yr)} | "
            f"{format_number(row.integrated_brier_score_1to5)} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The comparison uses the same temporal protocol as the 28 August "
        "experiment. Injury reporting is sparse before 2007 and changes over "
        "time, so any apparent gain must be described as incremental value from "
        "a public reporting proxy, not evidence from complete clinical records.",
        "",
        "The `history_zero_assumption` is never used because it depends on whether "
        "a player has an injury at any later career date. Missing values are "
        "handled inside each temporal training fold.",
        "",
    ])
    (OUTPUT_DIR / "VERIFIED_INJURY_FEATURE_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (OUTPUT_DIR / "verified_injury_feature_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("both", *CLASSIFICATION_TASK_FILES),
        default="both",
    )
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--skip-survival", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    final_module = load_final_experiment_module()
    config = final_module.load_uncertainty_config()
    source = pd.read_parquet(INJURY_PATH)
    injury_features = build_verified_injury_features(source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    injury_features.to_parquet(
        DATA_DIR / "player_season_verified_injury_features.parquet", index=False
    )

    tasks = tuple(CLASSIFICATION_TASK_FILES) if args.task == "both" else (args.task,)
    classification_results = {
        task: run_classification_task(
            final_module,
            task,
            injury_features,
            config,
            args.force_selection,
        )
        for task in tasks
    }
    reference = classification_results.get(
        "confirmation", next(iter(classification_results.values()))
    )
    survival_result = None
    if not args.skip_survival:
        survival_result = run_survival(
            final_module, injury_features, reference["feature_sets"]
        )

    manifest = {
        "status": "verified public injury feature comparison",
        "legacy_clinical_feature_sets_used": False,
        "legacy_synthetic_injury_file_used": False,
        "injury_source": str(INJURY_PATH),
        "injury_source_sha256": sha256_file(INJURY_PATH),
        "injury_feature_rows": int(len(injury_features)),
        "injury_features": verified_injury_feature_columns(),
        "zero_assumption": "profile_zero_assumption only",
        "history_zero_assumption_used": False,
        "imputation": "training-fold median only",
        "classification_tasks": {
            task: {
                "selected_feature_set": result["selected_feature_set"],
                "selected_models": {
                    str(key): value for key, value in result["selected_models"].items()
                },
                "selection_cache_signature": result["cache_signature"],
            }
            for task, result in classification_results.items()
        },
        "survival": None if survival_result is None else {
            "selected_feature_set": survival_result["selected_feature_set"],
            "selected_model": survival_result["selected_model"],
        },
        "feature_sets": reference["feature_sets"],
        "temporal_protocol": {
            "classification_selection_target_years": "2018-2021",
            "classification_locked_evaluation_target_years": "2022-2023",
            "survival_selection_origins": "2013-2014; outcomes censored at 2019",
            "survival_locked_origins": "2019-2020; outcomes censored at 2025",
        },
    }
    if survival_result is not None:
        write_report(classification_results, survival_result, manifest)
    else:
        (OUTPUT_DIR / "verified_injury_feature_comparison_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest["classification_tasks"], indent=2))


if __name__ == "__main__":
    main()
