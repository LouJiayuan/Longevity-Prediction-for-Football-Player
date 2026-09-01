from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "src" if (ROOT / "src").is_dir() else CODE_DIR
sys.path.insert(0, str(CODE_DIR))

from classification_targets import (
    landmark_is_eligible,
    target_within_calendar_horizon,
)


def load_weighted_minutes_module():
    path = CODE_DIR / "01_build_weighted_minutes.py"
    spec = importlib.util.spec_from_file_location("weighted_minutes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_sequential_application_module():
    path = CODE_DIR / "37_sequential_player_application.py"
    spec = importlib.util.spec_from_file_location("sequential_application", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def example_player_history():
    return pd.DataFrame({
        "season_start": [2010, 2011, 2012, 2013, 2014],
        "dropoff_candidate": [0, 0, 1, 1, 1],
        "dropoff_onset": [0, 0, 1, 0, 0],
        "dropoff_event": [0, 0, 0, 1, 0],
    })


def test_clubelo_is_strictly_before_match_date():
    module = load_weighted_minutes_module()
    sides = pd.DataFrame({
        "game_id": [1, 2],
        "club_id": [7, 7],
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
    })
    elo = pd.DataFrame({
        "club_id": [7, 7],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "Elo": [1500.0, 1510.0],
    })
    result = module.latest_prior_elo(sides, elo)
    assert result["asof_elo"].tolist() == [1500.0, 1510.0]
    assert result["elo_days_from_match"].tolist() == [1, 1]
    assert result["same_day_elo_available"].tolist() == [True, False]


def test_confirmation_and_early_warning_are_distinct_tasks():
    history = example_player_history()
    assert target_within_calendar_horizon(
        history, 1, 2, "confirmation"
    ) == 1
    assert target_within_calendar_horizon(
        history, 1, 1, "early_warning"
    ) == 1
    assert landmark_is_eligible(history, 2, "confirmation")
    assert not landmark_is_eligible(history, 2, "early_warning")


def test_early_warning_negative_requires_confirmation_season():
    history = example_player_history().iloc[:3].copy()
    history[["dropoff_candidate", "dropoff_onset", "dropoff_event"]] = 0
    assert pd.isna(target_within_calendar_horizon(
        history, 0, 2, "early_warning"
    ))
    assert target_within_calendar_horizon(
        example_player_history().assign(
            dropoff_candidate=0, dropoff_onset=0, dropoff_event=0
        ),
        0,
        2,
        "early_warning",
    ) == 0


def test_next_season_confirmation_starts_only_from_unconfirmed_low_season():
    module = load_sequential_application_module()
    histories = pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 3, 3],
        "season_start": [2010, 2011, 2012, 2010, 2011, 2010, 2012],
        "dropoff_candidate": [0, 1, 1, 1, 0, 1, 1],
        "dropoff_confirmed": [0, 0, 1, 0, 0, 0, 0],
        "dropoff_event": [0, 0, 1, 0, 0, 0, 0],
    })
    result = module.build_next_season_confirmation(histories)
    assert result[["player_id", "season_start", "target_end_season"]].to_dict(
        "records"
    ) == [
        {"player_id": 1, "season_start": 2011, "target_end_season": 2012},
        {"player_id": 2, "season_start": 2010, "target_end_season": 2011},
    ]
    assert result["_target"].tolist() == [1, 0]


def test_early_warning_curve_is_monotone_with_yearly_increments():
    module = load_sequential_application_module()
    frame = pd.DataFrame({
        "early_warning_risk_1yr": [0.20],
        "early_warning_risk_2yr": [0.18],
        "early_warning_risk_3yr": [0.45],
        "early_warning_risk_4yr": [0.40],
        "early_warning_risk_5yr": [0.70],
    })
    result = module.add_monotone_early_warning_curve(frame)
    assert result[[
        "early_warning_risk_1yr",
        "early_warning_risk_2yr",
        "early_warning_risk_3yr",
        "early_warning_risk_4yr",
        "early_warning_risk_5yr",
    ]].iloc[0].tolist() == [0.20, 0.20, 0.45, 0.45, 0.70]
    assert result[[
        "early_warning_risk_in_year_1",
        "early_warning_risk_in_year_2",
        "early_warning_risk_in_year_3",
        "early_warning_risk_in_year_4",
        "early_warning_risk_in_year_5",
    ]].iloc[0].round(6).tolist() == [0.20, 0.0, 0.25, 0.0, 0.25]
