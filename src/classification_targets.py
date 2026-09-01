"""Pure target builders for confirmation and early-warning classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


CLASSIFICATION_TASKS = ("confirmation", "early_warning")


def landmark_is_eligible(group: pd.DataFrame, row_index: int, task: str) -> bool:
    """Return whether a player-season can enter the requested prediction task."""
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"Unknown classification task: {task}")
    if group.iloc[: row_index + 1]["dropoff_event"].sum() > 0:
        return False
    if task == "early_warning" and int(
        group.iloc[row_index]["dropoff_candidate"]
    ) == 1:
        return False
    return True


def target_within_calendar_horizon(
    group: pd.DataFrame,
    row_index: int,
    horizon: int,
    task: str,
) -> float:
    """Return 1/0/NaN using task-specific events and censoring rules."""
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"Unknown classification task: {task}")
    current_season = int(group.iloc[row_index]["season_start"])
    event_column = "dropoff_event" if task == "confirmation" else "dropoff_onset"
    future_events = group[
        group[event_column].eq(1)
        & group["season_start"].gt(current_season)
        & group["season_start"].le(current_season + int(horizon))
    ]
    if not future_events.empty:
        return 1.0

    # Early warning needs the season after the horizon to verify that a low
    # season at the horizon boundary was not the onset of a confirmed event.
    extra_confirmation_season = 1 if task == "early_warning" else 0
    required_end = current_season + int(horizon) + extra_confirmation_season
    observed = set(group["season_start"].astype(int))
    fully_observed = all(
        season in observed
        for season in range(current_season + 1, required_end + 1)
    )
    return 0.0 if fully_observed else np.nan


def build_yearly_classification_dataset(
    player_seasons: pd.DataFrame,
    feature_columns: list[str],
    task: str,
    horizons=range(1, 6),
) -> pd.DataFrame:
    """Build one row per eligible landmark with cumulative horizon targets."""
    horizons = tuple(int(value) for value in horizons)
    rows = []
    for player_id, group in player_seasons.groupby("player_id"):
        group = group.sort_values("season_start").reset_index(drop=True)
        for row_index in range(len(group)):
            if not landmark_is_eligible(group, row_index, task):
                continue
            source = group.iloc[row_index]
            row = {column: source[column] for column in feature_columns}
            row["player_id"] = player_id
            row["season_start"] = source["season_start"]
            row["prediction_task"] = task
            row["complete_career_peak_cohort"] = source.get(
                "complete_career_peak_cohort", 1
            )
            row["career_history_uncertain"] = source.get(
                "career_history_uncertain", 0
            )
            for horizon in horizons:
                row[f"target_within_{horizon}yr"] = (
                    target_within_calendar_horizon(
                        group, row_index, horizon, task
                    )
                )
            observed = set(group["season_start"].astype(int))
            continuous_ahead = 0
            current_season = int(source["season_start"])
            while current_season + continuous_ahead + 1 in observed:
                continuous_ahead += 1
            row["seasons_ahead"] = max(
                0,
                min(max(horizons), continuous_ahead - (task == "early_warning")),
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    target_columns = [f"target_within_{value}yr" for value in horizons]
    if frame.empty:
        return frame
    return frame[frame[target_columns].notna().any(axis=1)].reset_index(drop=True)


def assert_cumulative_targets(frame: pd.DataFrame, horizons=range(1, 6)) -> None:
    """Validate that observed cumulative targets cannot fall from one to zero."""
    columns = [f"target_within_{int(value)}yr" for value in horizons]
    for left, right in zip(columns, columns[1:]):
        invalid = frame[left].eq(1) & frame[right].eq(0)
        if invalid.any():
            raise RuntimeError(
                f"Non-monotone cumulative targets between {left} and {right}: "
                f"{int(invalid.sum()):,} rows"
            )
