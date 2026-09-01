from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "src" if (ROOT / "src").is_dir() else CODE_DIR


def load_module():
    path = CODE_DIR / "00_align_longitudinal_injuries.py"
    spec = importlib.util.spec_from_file_location("longitudinal_injury_alignment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def example_event(**overrides):
    values = {
        "event_id": 1,
        "player_id": 10,
        "injury_start_date": pd.Timestamp("2020-06-25"),
        "resolved_end_date": pd.Timestamp("2020-07-05"),
        "reported_games_missed": 2.0,
        "resolved_calendar_days": 11,
        "identity_accepted": True,
        "event_validated": True,
        "long_spell_anomaly": False,
        "absence_category": "injury_or_other",
    }
    values.update(overrides)
    return values


def test_spell_is_split_at_july_season_boundary():
    module = load_module()
    segments = module.split_events_across_seasons(
        pd.DataFrame([example_event()]), 2019, 2020
    )
    assert segments["season_start"].tolist() == [2019, 2020]
    assert segments["overlap_days_raw"].tolist() == [6, 5]
    assert segments["event_started_this_season"].tolist() == [True, False]
    assert segments["validated_event_start"].tolist() == [1, 0]


def test_long_spell_is_retained_but_excluded_from_validated_days():
    module = load_module()
    segments = module.split_events_across_seasons(
        pd.DataFrame([
            example_event(
                injury_start_date=pd.Timestamp("2018-01-01"),
                resolved_end_date=pd.Timestamp("2021-01-01"),
                resolved_calendar_days=1097,
                event_validated=False,
                long_spell_anomaly=True,
            )
        ]),
        2017,
        2020,
    )
    assert len(segments) == 4
    assert segments["overlap_days_raw"].notna().all()
    assert segments["overlap_days_validated"].isna().all()
    assert segments["long_spell_anomaly_start"].sum() == 1


def test_no_report_is_unknown_until_an_assumption_is_selected():
    module = load_module()
    panel = pd.DataFrame({
        "player_id": pd.Series([10, 10], dtype="Int64"),
        "season_start": pd.Series([2019, 2020], dtype="Int64"),
        "name": ["Example", "Example"],
        "date_of_birth": pd.to_datetime(["1990-01-01", "1990-01-01"]),
    })
    profiles = pd.DataFrame({
        "player_id": pd.Series([10], dtype="Int64"),
        "player_name": ["Example (10)"],
        "source_date_of_birth": pd.to_datetime(["1990-01-01"]),
    })
    all_events = pd.DataFrame({"player_id": [10]})
    season_features = pd.DataFrame({
        "player_id": pd.Series([10], dtype="Int64"),
        "season_start": pd.Series([2020], dtype="Int64"),
        "reported_active_spell_count": [1],
        "reported_injury_start_count": [1],
        "validated_injury_start_count": [1],
        "validated_active_spell_count": [1],
        "validated_injury_days_sum": [10.0],
        "validated_unique_injury_days": [10],
        "severe_injury_start_count": [0],
        "major_injury_start_count": [0],
        "long_spell_anomaly_start_count": [0],
        "injury_or_other_start_count": [1],
        "illness_start_count": [0],
        "rest_or_fitness_start_count": [0],
        "unknown_reason_start_count": [0],
        "reported_games_missed_at_start": [2.0],
    })
    aligned = module.align_to_panel(panel, profiles, all_events, season_features)
    no_report = aligned.loc[aligned["season_start"].eq(2019)].iloc[0]
    assert np.isnan(no_report["reported_injury_start_count"])
    assert no_report["reported_injury_start_count_profile_zero_assumption"] == 0
    assert no_report["reported_injury_start_count_history_zero_assumption"] == 0
    assert not bool(no_report["missing_event_proves_zero_injury"])
