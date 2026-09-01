from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "src" if (ROOT / "src").is_dir() else CODE_DIR


def load_module():
    path = CODE_DIR / "38_verified_injury_feature_comparison.py"
    spec = importlib.util.spec_from_file_location("verified_injury_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source_fixture():
    rows = []
    for season, days, events in [(2019, 0.0, 0.0), (2020, 12.0, 1.0), (2021, 30.0, 1.0)]:
        rows.append({
            "player_id": 1,
            "season_start": season,
            "source_profile_present": True,
            "source_profile_dob_conflict": False,
            "source_player_has_any_injury_history": True,
            "injury_event_recorded_this_season": bool(events),
            "injury_main_window": True,
            "injury_sensitivity_window": False,
            "validated_unique_injury_days_profile_zero_assumption": days,
            "validated_injury_start_count_profile_zero_assumption": events,
            "severe_injury_start_count_profile_zero_assumption": float(days >= 28),
            "major_injury_start_count_profile_zero_assumption": 0.0,
            "reported_games_missed_at_start_profile_zero_assumption": events * 2,
            "validated_unique_injury_days_history_zero_assumption": days + 999,
        })
    return pd.DataFrame(rows)


def test_verified_features_do_not_use_future_history_zero_fields():
    module = load_module()
    source = source_fixture()
    result = module.build_verified_injury_features(source)
    assert not any("history_zero" in column for column in result.columns)
    assert result.loc[result["season_start"].eq(2020), "injury_days_current"].iloc[0] == 12
    assert result.loc[result["season_start"].eq(2020), "injury_days_lag1"].iloc[0] == 0
    assert result.loc[result["season_start"].eq(2020), "injury_days_2yr"].iloc[0] == 12


def test_future_event_does_not_change_earlier_cumulative_feature():
    module = load_module()
    source = source_fixture()
    baseline = module.build_verified_injury_features(source)
    changed = source.copy()
    changed.loc[changed["season_start"].eq(2021), [
        "validated_unique_injury_days_profile_zero_assumption",
        "validated_unique_injury_days_history_zero_assumption",
    ]] = [300.0, 1299.0]
    comparison = module.build_verified_injury_features(changed)
    columns = [
        "injury_days_current",
        "injury_days_lag1",
        "injury_days_2yr",
        "injury_career_days",
    ]
    left = baseline.loc[baseline["season_start"].le(2020), columns].to_numpy()
    right = comparison.loc[comparison["season_start"].le(2020), columns].to_numpy()
    assert np.array_equal(left, right, equal_nan=True)
