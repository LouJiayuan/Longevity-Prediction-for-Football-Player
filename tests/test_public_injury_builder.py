from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "src" if (ROOT / "src").is_dir() else CODE_DIR


def load_script(filename: str, module_name: str):
    path = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ambiguous_event_name_is_resolved_only_by_age():
    module = load_script("00_build_public_injury_data.py", "public_injury_builder")
    players = pd.DataFrame({
        "player_id": pd.Series([1, 2], dtype="Int64"),
        "name": ["Alex Smith", "Alex Smith"],
        "tm_player_name": ["Alex Smith", "Alex Smith"],
        "name_key": ["alexsmith", "alexsmith"],
        "tm_date_of_birth": pd.to_datetime(["1990-01-01", "2000-01-01"]),
    })
    source = pd.DataFrame({
        "Season": ["20/21"],
        "Injury": ["Hamstring injury"],
        "Days": ["14 days"],
        "Games missed": [2],
        "injury_from_parsed": ["2020-08-01"],
        "injury_until_parsed": ["2020-08-14"],
        "player_name": ["Alex Smith"],
        "player_age": [20],
        "club": ["Example FC"],
        "league": ["Example League"],
    })
    result = module.map_event_source(source, players)
    assert result.loc[0, "player_id"] == 2
    assert result.loc[0, "mapping_method"] == "ambiguous_name_resolved_by_age"
    assert result.loc[0, "mapping_confidence"] == "high"


def test_aggregate_duplicate_rows_are_not_summed():
    module = load_script("00_build_public_injury_data.py", "public_injury_aggregate")
    players = pd.DataFrame({
        "player_id": pd.Series([10], dtype="Int64"),
        "name": ["Aaron Connolly"],
        "tm_player_name": ["Aaron Connolly"],
        "name_key": ["aaronconnolly"],
        "tm_date_of_birth": pd.to_datetime(["2000-01-28"]),
    })
    source = pd.DataFrame({
        "p_id2": ["aaronconnolly", "aaronconnolly"],
        "dob": ["2000-01-28", "2000-01-28"],
        "start_year": [2019, 2019],
        "season_days_injured": [13, 13],
        "total_days_injured": [161, 161],
    })
    result = module.map_aggregate_source(source, players)
    assert len(result) == 1
    assert result.loc[0, "season_days_injured"] == 13
    assert result.loc[0, "source_duplicate_rows"] == 2
    assert not bool(result.loc[0, "source_conflict"])


def test_transfermarkt_injury_page_parser():
    module = load_script(
        "00_collect_transfermarkt_injury_history.py", "tm_injury_collector"
    )
    html = """
    <table class="items">
      <thead><tr><th>Season</th><th>Injury</th><th>from</th><th>until</th>
      <th>Days</th><th>Games missed</th></tr></thead>
      <tbody><tr><td>22/23</td><td>Ankle injury</td><td>21/01/2023</td>
      <td>28/01/2023</td><td>8 days</td><td>2</td></tr></tbody>
    </table>
    """
    result = module.parse_injury_page(html, 99, "Test Player", "https://example.test")
    assert len(result) == 1
    assert result.loc[0, "season_start"] == 2022
    assert result.loc[0, "days_lost"] == 8
    assert result.loc[0, "games_missed"] == 2
