"""Evaluate raw outcome, peak-threshold, and alpha/beta definition sensitivity.

This script keeps the existing 2000-2025 match universe and temporal protocol,
but rebuilds the *outcome definition* when required.

Outputs are deliberately separate from the official experiment:

- a raw-minute label/outcome version;
- adjusted-minute labels at 50%, 60%, 65%, and 70% of prior peak;
- a 5 x 5 alpha/beta grid, where both adjusted exposure and label are rebuilt.

Every model uses the matched ``age_plus_minutes`` trajectory feature ladder and
the final XGBoost hyperparameters. AUC/AP are reported without post-hoc model
selection or calibration; their role is sensitivity, not model selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from modeling_common import clean_feature_matrix, enforce_probability_bounds  # noqa: E402
from uncertainty_common import add_target_end_season, load_uncertainty_config  # noqa: E402


KIMI_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = KIMI_DIR / "data" / "processed"
OUTPUT_DIR = KIMI_DIR / "outputs"

HORIZONS = (2, 5)
PEAK_RATIOS = (0.50, 0.60, 0.65, 0.70)
ALPHAS = (0.15, 0.20, 0.25, 0.30, 0.35)
BETAS = (0.04, 0.06, 0.08, 0.10, 0.12)
CANONICAL_ALPHA = 0.25
CANONICAL_BETA = 0.08
CANONICAL_RATIO = 0.65
MIN_HISTORY_SEASONS = 2

STABLE_COLUMNS = [
    "player_id",
    "season_start",
    "total_minutes",
    "final_weighted_minutes",
    "age",
    "age_sq",
    "games_played",
    "previous_season_observed",
    "career_games_played",
    "career_seasons",
    "dropoff_event",
]
COMMON_FEATURES = [
    "age",
    "age_sq",
    "games_played",
    "previous_season_observed",
    "career_games_played",
    "career_seasons",
]
TRAJECTORY_FEATURES = [
    "scenario_minutes",
    "scenario_minutes_lag1",
    "scenario_minutes_lag2",
    "scenario_minutes_avg2yr",
    "scenario_minutes_yoy",
    "career_scenario_minutes",
    "career_scenario_minutes_per_season",
]
MODEL_FEATURES = COMMON_FEATURES + TRAJECTORY_FEATURES


def xgboost_classifier() -> xgb.XGBClassifier:
    """Final-experiment XGBoost settings, fixed for every sensitivity cell."""
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


def build_labels(frame: pd.DataFrame, peak_ratio: float) -> pd.DataFrame:
    """Create strict-prior-peak, two-adjacent-season labels for scenario minutes."""
    out = frame.sort_values(["player_id", "season_start"]).reset_index()
    out = out.rename(columns={"index": "source_index"})
    n_rows = len(out)
    history = np.zeros(n_rows, dtype=int)
    prior_peak = np.full(n_rows, np.nan)
    prior_peak_season = np.full(n_rows, np.nan)

    for positions in out.groupby("player_id", sort=False).groups.values():
        positions = np.asarray(list(positions), dtype=int)
        minutes = out.loc[positions, "scenario_minutes"].to_numpy(dtype=float)
        seasons = out.loc[positions, "season_start"].to_numpy(dtype=float)
        best: int | None = None
        for local_position, row_position in enumerate(positions):
            history[row_position] = local_position
            if best is not None:
                prior_peak[row_position] = minutes[best]
                prior_peak_season[row_position] = seasons[best]
            # Keep the most recent season in a tied peak, matching script 02.
            if best is None or minutes[local_position] >= minutes[best]:
                best = local_position

    out["history_seasons_before"] = history
    out["scenario_prior_peak_minutes"] = prior_peak
    out["scenario_prior_peak_season"] = prior_peak_season
    out["scenario_minutes_vs_prior_peak"] = (
        out["scenario_minutes"] / out["scenario_prior_peak_minutes"].replace(0, np.nan)
    )
    eligible = (
        out["history_seasons_before"].ge(MIN_HISTORY_SEASONS)
        & out["scenario_prior_peak_minutes"].gt(0)
    )
    candidate = eligible & out["scenario_minutes_vs_prior_peak"].lt(peak_ratio)
    previous_candidate = candidate.groupby(out["player_id"]).shift(
        1, fill_value=False
    )
    previous_season = out.groupby("player_id")["season_start"].shift(1)
    confirmed = (
        candidate
        & previous_candidate
        & out["season_start"].eq(previous_season + 1)
    )
    previous_confirmation = confirmed.groupby(out["player_id"]).transform(
        lambda values: values.shift(1, fill_value=False).cummax()
    )
    event = confirmed & ~previous_confirmation
    onset = event.groupby(out["player_id"]).shift(-1, fill_value=False)

    out["scenario_candidate"] = candidate.astype(np.int8)
    out["scenario_confirmed"] = confirmed.astype(np.int8)
    out["scenario_event"] = event.astype(np.int8)
    out["scenario_onset"] = onset.astype(np.int8)
    return out.set_index("source_index").sort_index()


def add_trajectory_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive prospective raw/adjusted trajectory features from scenario minutes."""
    out = frame.sort_values(["player_id", "season_start"]).copy()
    previous_season = out.groupby("player_id")["season_start"].shift(1)
    two_seasons_back = out.groupby("player_id")["season_start"].shift(2)
    adjacent_lag1 = out["season_start"].eq(previous_season + 1)
    adjacent_lag2 = out["season_start"].eq(two_seasons_back + 2) & adjacent_lag1
    lag1 = out.groupby("player_id")["scenario_minutes"].shift(1)
    lag2 = out.groupby("player_id")["scenario_minutes"].shift(2)
    out["scenario_minutes_lag1"] = lag1.where(adjacent_lag1)
    out["scenario_minutes_lag2"] = lag2.where(adjacent_lag2)
    out["scenario_minutes_avg2yr"] = pd.concat(
        [out["scenario_minutes_lag1"], out["scenario_minutes_lag2"]], axis=1
    ).mean(axis=1, skipna=True)
    out["scenario_minutes_yoy"] = (
        out["scenario_minutes"] / out["scenario_minutes_lag1"].clip(lower=1) - 1
    )
    out["career_scenario_minutes"] = out.groupby("player_id")[
        "scenario_minutes"
    ].cumsum()
    out["career_scenario_minutes_per_season"] = (
        out["career_scenario_minutes"] / out["career_seasons"].clip(lower=1)
    )
    return out


def add_horizon_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Build official-style future targets from a scenario-specific first event."""
    out = frame.sort_values(["player_id", "season_start"]).copy()
    for horizon in HORIZONS:
        out[f"target_within_{horizon}yr"] = np.nan

    for positions in out.groupby("player_id", sort=False).groups.values():
        positions = np.asarray(list(positions), dtype=int)
        seasons = out.loc[positions, "season_start"].to_numpy(dtype=int)
        events = out.loc[positions, "scenario_event"].to_numpy(dtype=int)
        observed = set(seasons)
        event_seasons = seasons[events == 1]
        event_seen = False
        for local_position, row_position in enumerate(positions):
            if event_seen or events[local_position] == 1:
                event_seen = True
                continue
            current_season = int(seasons[local_position])
            for horizon in HORIZONS:
                future_event = np.any(
                    (event_seasons > current_season)
                    & (event_seasons <= current_season + horizon)
                )
                target_column = f"target_within_{horizon}yr"
                if future_event:
                    out.loc[row_position, target_column] = 1
                elif all(
                    season in observed
                    for season in range(
                        current_season + 1, current_season + horizon + 1
                    )
                ):
                    out.loc[row_position, target_column] = 0
    return out


def first_event_seasons(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.loc[frame["scenario_event"].eq(1)]
        .groupby("player_id")["season_start"]
        .min()
    )


def label_diagnostics(
    frame: pd.DataFrame, canonical_event: pd.Series
) -> dict[str, float | int]:
    """Measure how a scenario label changes relative to canonical adjusted 65%."""
    scenario_event = frame["scenario_event"].astype(int)
    canonical_event = canonical_event.reindex(frame.index).astype(int)
    players = pd.Index(frame["player_id"].unique())
    canonical_first = (
        frame.assign(_event=canonical_event)
        .loc[lambda value: value["_event"].eq(1)]
        .groupby("player_id")["season_start"]
        .min()
        .reindex(players)
        .fillna(-1)
    )
    scenario_first = first_event_seasons(frame).reindex(players).fillna(-1)
    return {
        "candidate_rows": int(frame["scenario_candidate"].sum()),
        "confirmed_rows": int(frame["scenario_confirmed"].sum()),
        "first_confirmed_events": int(scenario_event.sum()),
        "event_row_flip_count_vs_canonical": int(
            scenario_event.ne(canonical_event).sum()
        ),
        "event_row_flip_rate_vs_canonical": float(
            scenario_event.ne(canonical_event).mean()
        ),
        "first_event_season_change_players_vs_canonical": int(
            scenario_first.ne(canonical_first).sum()
        ),
        "first_event_season_change_rate_vs_canonical": float(
            scenario_first.ne(canonical_first).mean()
        ),
    }


def evaluate_scenario(
    frame: pd.DataFrame, scenario: dict[str, object], config: dict
) -> list[dict[str, float | int | str]]:
    """Fit the fixed core classifier using the current scenario's own outcome."""
    records = []
    for horizon in HORIZONS:
        target_column = f"target_within_{horizon}yr"
        data = frame.dropna(subset=[target_column]).copy()
        data["_target"] = data[target_column].astype(int)
        data = add_target_end_season(data, horizon)
        train = data[
            data["target_end_season"].le(config["selection_target_year_end"])
        ].copy()
        test = data[
            data["target_end_season"].between(
                config["evaluation_target_year_start"],
                config["evaluation_target_year_end"],
            )
        ].copy()
        if train.empty or test.empty or train["_target"].nunique() < 2:
            raise RuntimeError(
                f"Insufficient temporal data for {scenario['scenario_id']} {horizon}y."
            )
        X_train, X_test, _, used, _ = clean_feature_matrix(
            train, test, test, MODEL_FEATURES
        )
        model = xgboost_classifier()
        model.fit(X_train, train["_target"].astype(int))
        probability = enforce_probability_bounds(model.predict_proba(X_test)[:, 1])
        records.append({
            **scenario,
            "horizon_years": horizon,
            "feature_set": "matched_age_plus_minutes_trajectory",
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


def scenario_frame(
    base: pd.DataFrame, minutes: pd.Series, peak_ratio: float
) -> pd.DataFrame:
    """Attach scenario exposure, labels, trajectories, and future targets."""
    if not minutes.index.equals(base.index):
        minutes = minutes.reindex(base.index)
    if minutes.isna().any():
        raise RuntimeError("Scenario adjusted minutes are missing player-seasons.")
    frame = base.copy()
    frame["scenario_minutes"] = minutes.to_numpy(dtype=float)
    frame = build_labels(frame, peak_ratio)
    frame = add_trajectory_features(frame)
    return add_horizon_targets(frame)


def aggregate_adjusted_minutes(
    match_keys: pd.DataFrame,
    minutes_played: np.ndarray,
    quality_signal: np.ndarray,
    opponent_signal: np.ndarray,
    alpha: float,
    beta: float,
    bounds: tuple[float, float],
) -> pd.Series:
    """Recalculate a player-season exposure for an alpha/beta cell."""
    multiplier = np.clip(
        (1.0 + beta * quality_signal) * (1.0 + alpha * opponent_signal),
        *bounds,
    )
    aggregate = match_keys.copy()
    aggregate["scenario_minutes"] = minutes_played * multiplier
    series = aggregate.groupby(["player_id", "season_start"], sort=False)[
        "scenario_minutes"
    ].sum()
    return series


def alignment_series(base: pd.DataFrame, aggregate: pd.Series) -> pd.Series:
    index = pd.MultiIndex.from_frame(base[["player_id", "season_start"]])
    values = aggregate.reindex(index)
    if values.isna().any():
        raise RuntimeError(
            f"Missing match aggregation for {int(values.isna().sum()):,} player-seasons."
        )
    return pd.Series(values.to_numpy(dtype=float), index=base.index)


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, divider]
    for row in frame.itertuples(index=False):
        values = []
        for column, _ in columns:
            value = getattr(row, column)
            if isinstance(value, float):
                if column in {"auc", "average_precision", "test_event_rate"}:
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_uncertainty_config()
    base = pd.read_parquet(
        PROCESSED_DIR / "player_season_features.parquet", columns=STABLE_COLUMNS
    ).sort_values(["player_id", "season_start"]).reset_index(drop=True)
    if base[STABLE_COLUMNS].isna().all(axis=None):
        raise RuntimeError("Player-season source could not be read.")

    # Canonical 65% labels must be reproduced before any sensitivity result is used.
    canonical_frame = scenario_frame(
        base, base["final_weighted_minutes"], CANONICAL_RATIO
    )
    canonical_event = canonical_frame["scenario_event"].astype(int)
    if not canonical_event.equals(base["dropoff_event"].astype(int)):
        mismatches = int(canonical_event.ne(base["dropoff_event"].astype(int)).sum())
        raise RuntimeError(
            f"Canonical 65% label recreation failed: {mismatches:,} event mismatches."
        )

    raw_frame = scenario_frame(base, base["total_minutes"], CANONICAL_RATIO)
    raw_output = raw_frame[[
        "player_id", "season_start", "total_minutes", "scenario_prior_peak_minutes",
        "scenario_prior_peak_season", "scenario_minutes_vs_prior_peak",
        "scenario_candidate", "scenario_confirmed", "scenario_event", "scenario_onset",
    ]].rename(columns={
        "scenario_prior_peak_minutes": "raw_prior_peak_minutes",
        "scenario_prior_peak_season": "raw_prior_peak_season",
        "scenario_minutes_vs_prior_peak": "raw_minutes_vs_prior_peak",
        "scenario_candidate": "raw_dropoff_candidate",
        "scenario_confirmed": "raw_dropoff_confirmed",
        "scenario_event": "raw_dropoff_event",
        "scenario_onset": "raw_dropoff_onset",
    })
    raw_output.to_parquet(
        PROCESSED_DIR / "player_season_raw_minutes_labeled.parquet", index=False
    )

    canonical_diagnostics = label_diagnostics(canonical_frame, canonical_event)
    raw_diagnostics = label_diagnostics(raw_frame, canonical_event)
    outcome_rows = []
    outcome_rows.extend(evaluate_scenario(
        raw_frame,
        {
            "scenario_family": "outcome_definition",
            "scenario_id": "raw_minutes_peak_65",
            "minutes_definition": "raw_minutes",
            "peak_ratio": CANONICAL_RATIO,
            "alpha": np.nan,
            "beta": np.nan,
            **raw_diagnostics,
        },
        config,
    ))

    threshold_event_counts = []
    for ratio in PEAK_RATIOS:
        print(f"Evaluating adjusted threshold={ratio:.2f}...", flush=True)
        frame = canonical_frame if ratio == CANONICAL_RATIO else scenario_frame(
            base, base["final_weighted_minutes"], ratio
        )
        diagnostics = label_diagnostics(frame, canonical_event)
        threshold_event_counts.append(diagnostics["first_confirmed_events"])
        outcome_rows.extend(evaluate_scenario(
            frame,
            {
                "scenario_family": "outcome_definition",
                "scenario_id": f"adjusted_peak_{int(ratio * 100):02d}",
                "minutes_definition": "adjusted_minutes_canonical_alpha_beta",
                "peak_ratio": ratio,
                "alpha": CANONICAL_ALPHA,
                "beta": CANONICAL_BETA,
                **diagnostics,
            },
            config,
        ))
    if not np.all(np.diff(np.asarray(threshold_event_counts)) >= 0):
        raise RuntimeError("Threshold event counts must be non-decreasing.")
    outcome_metrics = pd.DataFrame(outcome_rows)

    # Rebuild exposure for every alpha/beta combination directly from match rows.
    match_columns = [
        "player_id", "season_start", "minutes_played", "match_quality_elo",
        "top5_reference_elo", "opponent_expected_score",
    ]
    matches = pd.read_parquet(
        PROCESSED_DIR / "match_level_weighted_minutes.parquet", columns=match_columns
    )
    quality_scale = float(json.loads(
        (KIMI_DIR / "config" / "match_quality_experiment.json").read_text(
            encoding="utf-8"
        )
    )["quality_elo_scale"])
    bounds = tuple(json.loads(
        (KIMI_DIR / "config" / "match_quality_experiment.json").read_text(
            encoding="utf-8"
        )
    )["final_multiplier_bounds"])
    match_keys = matches[["player_id", "season_start"]]
    minutes_played = matches["minutes_played"].to_numpy(dtype=float)
    quality_signal = np.tanh(
        (matches["match_quality_elo"].to_numpy(dtype=float)
         - matches["top5_reference_elo"].to_numpy(dtype=float)) / quality_scale
    )
    opponent_signal = 2.0 * matches["opponent_expected_score"].to_numpy(dtype=float) - 1.0

    canonical_aggregate = aggregate_adjusted_minutes(
        match_keys, minutes_played, quality_signal, opponent_signal,
        CANONICAL_ALPHA, CANONICAL_BETA, bounds,
    )
    canonical_aggregate = alignment_series(base, canonical_aggregate)
    max_error = float(np.abs(
        canonical_aggregate - base["final_weighted_minutes"]
    ).max())
    if max_error > 1e-6:
        raise RuntimeError(
            f"Canonical match aggregation mismatch: max error={max_error}."
        )

    alpha_beta_rows = []
    for alpha in ALPHAS:
        for beta in BETAS:
            print(f"Evaluating alpha={alpha:.2f}, beta={beta:.2f}...", flush=True)
            minutes = alignment_series(
                base,
                aggregate_adjusted_minutes(
                    match_keys, minutes_played, quality_signal, opponent_signal,
                    alpha, beta, bounds,
                ),
            )
            frame = canonical_frame if (
                alpha == CANONICAL_ALPHA and beta == CANONICAL_BETA
            ) else scenario_frame(base, minutes, CANONICAL_RATIO)
            diagnostics = label_diagnostics(frame, canonical_event)
            alpha_beta_rows.extend(evaluate_scenario(
                frame,
                {
                    "scenario_family": "alpha_beta",
                    "scenario_id": f"alpha_{alpha:.2f}_beta_{beta:.2f}",
                    "minutes_definition": "adjusted_minutes_scenario_specific",
                    "peak_ratio": CANONICAL_RATIO,
                    "alpha": alpha,
                    "beta": beta,
                    "canonical_minutes_max_abs_error": max_error if (
                        alpha == CANONICAL_ALPHA and beta == CANONICAL_BETA
                    ) else np.nan,
                    **diagnostics,
                },
                config,
            ))
    alpha_beta_metrics = pd.DataFrame(alpha_beta_rows)

    outcome_path = OUTPUT_DIR / "outcome_definition_sensitivity_metrics.csv"
    alpha_beta_path = OUTPUT_DIR / "alpha_beta_auc_sensitivity_metrics.csv"
    manifest_path = OUTPUT_DIR / "outcome_definition_sensitivity_manifest.json"
    report_path = OUTPUT_DIR / "outcome_definition_sensitivity_report.md"
    outcome_metrics.to_csv(outcome_path, index=False)
    alpha_beta_metrics.to_csv(alpha_beta_path, index=False)
    manifest = {
        "primary_outcome": "adjusted minutes, 65% of strict prior peak, two adjacent seasons",
        "raw_outcome": "raw minutes, 65% of strict prior peak, two adjacent seasons",
        "model": "XGBoost final-experiment hyperparameters",
        "feature_set": "matched age_plus_minutes trajectory; no label/peak ratio feature",
        "horizons": list(HORIZONS),
        "thresholds": list(PEAK_RATIOS),
        "alphas": list(ALPHAS),
        "betas": list(BETAS),
        "evaluation_target_end_seasons": [
            config["evaluation_target_year_start"],
            config["evaluation_target_year_end"],
        ],
        "imputation": "training-fold median only",
        "note": "AUC/AP across different outcome definitions describe sensitivity; they must not be used to select a threshold or alpha/beta post hoc.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    outcome_columns = [
        ("scenario_id", "Scenario"), ("horizon_years", "Horizon"),
        ("first_confirmed_events", "Events"), ("n_test", "Test n"),
        ("test_event_rate", "Event rate"), ("auc", "AUC"),
        ("average_precision", "AP"),
    ]
    alpha_beta_ranges = alpha_beta_metrics.groupby("horizon_years").agg(
        auc_min=("auc", "min"),
        auc_max=("auc", "max"),
        ap_min=("average_precision", "min"),
        ap_max=("average_precision", "max"),
        event_min=("first_confirmed_events", "min"),
        event_max=("first_confirmed_events", "max"),
        label_flip_max=("event_row_flip_rate_vs_canonical", "max"),
    ).reset_index()
    report = f"""# Outcome Definition and Weight-Parameter Sensitivity

## What changes in this experiment

The official result predicts adjusted-minute drop-off at 65% of the strictly
prior peak, confirmed by two adjacent low seasons. This sensitivity analysis
adds a **separate raw-minute outcome** and rebuilds adjusted labels whenever a
peak ratio, alpha or beta changes. Therefore each AUC is conditional on that
scenario's own ground truth; a higher AUC does not make one threshold more
correct.

All scenarios use the same 2/5-year target-end temporal evaluation window,
the matched `age_plus_minutes` trajectory feature ladder, final XGBoost
hyperparameters and training-fold-only median imputation. Neither a peak ratio
nor a drop-off label is used as a predictor.

## Raw and threshold outcome versions

{markdown_table(outcome_metrics.sort_values(['scenario_id', 'horizon_years']), outcome_columns)}

## Alpha/beta AUC ranges across the 25 cells

{markdown_table(alpha_beta_ranges, [
    ('horizon_years', 'Horizon'), ('auc_min', 'Minimum AUC'),
    ('auc_max', 'Maximum AUC'), ('ap_min', 'Minimum AP'),
    ('ap_max', 'Maximum AP'), ('event_min', 'Minimum events'),
    ('event_max', 'Maximum events'), ('label_flip_max', 'Maximum event-row flip rate'),
])}

The full 25-cell x 2-horizon table is in `{alpha_beta_path.name}`.

## Interpretation boundary

This is a definition sensitivity analysis, not a model-selection competition.
Changing 65%, alpha or beta changes event prevalence, membership of the
at-risk cohort and sometimes the future target itself. Report the stability
range and label-change rates; do not choose the parameter value with the
largest observed AUC after inspecting this table.
"""
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    for path in [outcome_path, alpha_beta_path, manifest_path, report_path]:
        print(f"Saved: {path}")
    print(
        "Saved: "
        f"{PROCESSED_DIR / 'player_season_raw_minutes_labeled.parquet'}"
    )


if __name__ == "__main__":
    main()
