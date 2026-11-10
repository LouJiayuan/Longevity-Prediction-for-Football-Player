"""
Build match-quality adjusted minutes for senior club football.

The canonical competition-tier multiplier is replaced by two match-level Elo
components:

1. absolute match quality from the mean Elo of both clubs;
2. relative opponent difficulty from the Elo gap.

Club league rank is retained as a model feature but is never multiplied into
adjusted minutes. National-team, youth, reserve, friendly, and unclassified
extension competitions are excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
# data/ 默认与代码目录同级；不存在时回退到上一级目录。
DATA_ROOT = BASE_DIR / "data" if (BASE_DIR / "data").is_dir() else PROJECT_DIR / "data"
DATA_DIR = DATA_ROOT
CANONICAL_DIR = DATA_DIR / "transfermarkt_public"
EXTENSION_DIR = (
    DATA_DIR / "external" / "transfermarkt_two_tiers_2000_2025"
)
OUT_DIR = DATA_DIR / "processed"
REPORT_DIR = BASE_DIR / "outputs"
CONFIG_PATH = BASE_DIR / "config" / "match_quality_experiment.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def season_start(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return pd.Series(
        np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1),
        index=values.index,
        dtype="Int64",
    )


def competition_family(comp_type: str, sub_type: str) -> str:
    comp_type = str(comp_type or "")
    sub_type = str(sub_type or "").lower()
    if comp_type == "domestic_league":
        return "domestic_league"
    if comp_type == "domestic_cup":
        return "domestic_cup"
    if "qualifying" in sub_type:
        return "continental_qualifier"
    if "super_cup" in sub_type:
        return "super_cup"
    if "play_off" in sub_type:
        return "playoff"
    if (
        comp_type == "international_cup"
        or "champions_league" in sub_type
        or "europa_league" in sub_type
        or "conference_league" in sub_type
    ):
        return "continental_cup"
    return "other_senior_club"


def build_competition_scope(
    competitions: pd.DataFrame, config: dict
) -> pd.DataFrame:
    excluded_types = set(config["exclude_competition_types"])
    canonical = competitions[
        ~competitions["type"].isin(excluded_types)
    ].copy()
    canonical["competition_family"] = canonical.apply(
        lambda row: competition_family(row["type"], row["sub_type"]), axis=1
    )
    canonical["tier"] = np.where(
        canonical["sub_type"].eq("first_tier"), 1, np.nan
    )
    canonical["scope_source"] = "canonical_metadata"
    canonical["include_in_adjusted_minutes"] = True

    custom = pd.DataFrame(config["custom_competitions"])
    custom["scope_source"] = "audited_extension"
    custom["include_in_adjusted_minutes"] = True
    custom["type"] = custom["competition_family"].replace({
        "domestic_league": "domestic_league",
        "domestic_cup": "domestic_cup",
    })
    custom["sub_type"] = custom["competition_family"]

    columns = [
        "competition_id",
        "name",
        "type",
        "sub_type",
        "country_name",
        "competition_family",
        "tier",
        "scope_source",
        "include_in_adjusted_minutes",
    ]
    scope = pd.concat(
        [canonical[columns], custom[columns]], ignore_index=True
    )
    scope = scope.drop_duplicates("competition_id", keep="last")
    return scope.sort_values("competition_id").reset_index(drop=True)


def load_canonical_matches(
    scope: pd.DataFrame, start_season: int, end_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(CANONICAL_DIR / "games.csv", low_memory=False)
    appearances = pd.read_csv(
        CANONICAL_DIR / "appearances.csv", low_memory=False
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games["season_start"] = pd.to_numeric(
        games["season"], errors="coerce"
    ).astype("Int64")
    games = games.merge(
        scope[[
            "competition_id",
            "competition_family",
            "include_in_adjusted_minutes",
        ]],
        on="competition_id",
        how="left",
    )
    games = games[
        games["include_in_adjusted_minutes"].eq(True)
        & games["season_start"].between(start_season, end_season)
    ].copy()

    game_cols = [
        "game_id",
        "date",
        "season_start",
        "competition_id",
        "competition_family",
        "home_club_id",
        "away_club_id",
    ]
    matches = appearances.drop(columns=["date"], errors="ignore").merge(
        games[game_cols], on="game_id", how="inner", validate="many_to_one"
    )
    if "competition_id_x" in matches:
        mismatch = matches["competition_id_x"].ne(
            matches["competition_id_y"]
        )
        if mismatch.any():
            raise RuntimeError(
                f"Canonical appearance/game competition mismatch: "
                f"{int(mismatch.sum()):,}"
            )
        matches = matches.drop(columns=["competition_id_x"]).rename(
            columns={"competition_id_y": "competition_id"}
        )
    matches["opponent_id"] = np.where(
        matches["player_club_id"].eq(matches["home_club_id"]),
        matches["away_club_id"],
        matches["home_club_id"],
    )
    matches["source_dataset"] = "canonical"
    matches["venue"] = np.where(
        matches["player_club_id"].eq(matches["home_club_id"]),
        "home",
        "away",
    )
    return matches, games


def load_extension_matches(
    scope: pd.DataFrame,
    canonical_ids: set[str],
    start_season: int,
    end_season: int,
    canonical_start_season: int,
) -> pd.DataFrame:
    custom_ids = set(
        scope.loc[
            scope["scope_source"].eq("audited_extension")
            & scope["competition_family"].eq("domestic_league"),
            "competition_id",
        ]
    ) - canonical_ids
    early_ids = set(
        scope.loc[
            scope["include_in_adjusted_minutes"].eq(True), "competition_id"
        ]
    )
    path = (
        EXTENSION_DIR
        / "career_appearances"
        / "career_club_appearances.parquet"
    )
    columns = [
        "appearance_id",
        "game_id",
        "player_id",
        "player_name",
        "season",
        "competition_id",
        "date",
        "player_club_id",
        "opponent_club_id",
        "venue",
        "minutes_played",
        "yellow_cards",
        "goals",
        "assists",
    ]
    frames = []
    early_end = min(end_season, canonical_start_season - 1)
    if start_season <= early_end:
        frames.append(pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("season", ">=", start_season),
                ("season", "<=", early_end),
                ("competition_id", "in", sorted(early_ids)),
            ],
        ))
    custom_start = max(start_season, canonical_start_season)
    if custom_ids and custom_start <= end_season:
        frames.append(pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("season", ">=", custom_start),
                ("season", "<=", end_season),
                ("competition_id", "in", sorted(custom_ids)),
            ],
        ))
    if not frames:
        return pd.DataFrame(columns=columns)
    matches = pd.concat(frames, ignore_index=True)
    matches["date"] = pd.to_datetime(
        matches["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    matches["season_start"] = matches["season"].astype("Int64")
    matches["competition_id"] = matches["competition_id"].replace(
        {"EFD2": "GB2"}
    )
    family_map = scope.set_index("competition_id")[
        "competition_family"
    ].to_dict()
    matches["competition_family"] = matches["competition_id"].map(family_map)
    matches = matches.rename(
        columns={
            "opponent_club_id": "opponent_id",
            "player_name": "player_name_extension",
        }
    )
    matches["player_current_club_id"] = pd.NA
    matches["red_cards"] = pd.NA
    matches["source_dataset"] = "two_tier_career_extension"
    matches["home_club_id"] = np.where(
        matches["venue"].eq("home"),
        matches["player_club_id"],
        np.where(
            matches["venue"].eq("away"),
            matches["opponent_id"],
            pd.NA,
        ),
    )
    matches["away_club_id"] = np.where(
        matches["venue"].eq("away"),
        matches["player_club_id"],
        np.where(
            matches["venue"].eq("home"),
            matches["opponent_id"],
            pd.NA,
        ),
    )
    matches = matches.rename(
        columns={"player_name_extension": "player_name"}
    )
    return matches


def build_club_season_competitions(
    matches: pd.DataFrame,
    in_scope_domestic_leagues: set[str],
) -> pd.DataFrame:
    league = matches[
        matches["competition_family"].eq("domestic_league")
        & matches["competition_id"].isin(in_scope_domestic_leagues)
    ][[
        "game_id",
        "season_start",
        "competition_id",
        "player_club_id",
        "opponent_id",
    ]].copy()
    player_sides = league.rename(
        columns={"player_club_id": "club_id"}
    )[["game_id", "season_start", "competition_id", "club_id"]]
    opponent_sides = league.rename(
        columns={"opponent_id": "club_id"}
    )[["game_id", "season_start", "competition_id", "club_id"]]
    sides = pd.concat([player_sides, opponent_sides], ignore_index=True)
    sides = sides.dropna(subset=["club_id"]).drop_duplicates(
        ["game_id", "club_id"]
    )
    counts = (
        sides.groupby(
            ["club_id", "season_start", "competition_id"], as_index=False
        )
        .size()
        .sort_values(
            ["club_id", "season_start", "size"],
            ascending=[True, True, False],
        )
    )
    mapping = counts.drop_duplicates(
        ["club_id", "season_start"], keep="first"
    ).rename(columns={"competition_id": "domestic_competition_id"})
    return mapping[
        ["club_id", "season_start", "domestic_competition_id", "size"]
    ]


def attach_competition_context(
    matches: pd.DataFrame, club_seasons: pd.DataFrame
) -> pd.DataFrame:
    player_map = club_seasons.rename(
        columns={
            "club_id": "player_club_id",
            "domestic_competition_id": "player_domestic_comp",
        }
    )[["player_club_id", "season_start", "player_domestic_comp"]]
    opponent_map = club_seasons.rename(
        columns={
            "club_id": "opponent_id",
            "domestic_competition_id": "opponent_domestic_comp",
        }
    )[["opponent_id", "season_start", "opponent_domestic_comp"]]
    return matches.merge(
        player_map,
        on=["player_club_id", "season_start"],
        how="left",
        validate="many_to_one",
    ).merge(
        opponent_map,
        on=["opponent_id", "season_start"],
        how="left",
        validate="many_to_one",
    )


def filter_audited_extension_seasons(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    coverage_path = (
        EXTENSION_DIR / "coverage" / "two_tier_coverage_by_season.csv"
    )
    if not coverage_path.exists():
        raise FileNotFoundError(
            f"Run the two-tier coverage analyzer first: {coverage_path}"
        )
    coverage = pd.read_csv(coverage_path)
    eligible = coverage[coverage["training_eligible"].eq(True)][[
        "season", "logical_competition_id"
    ]].drop_duplicates()
    eligible["_audited_extension_season"] = True
    audited = matches.merge(
        eligible.rename(
            columns={
                "season": "season_start",
                "logical_competition_id": "player_domestic_comp",
            }
        ),
        on=["season_start", "player_domestic_comp"],
        how="left",
        validate="many_to_one",
    )
    extension = audited["source_dataset"].eq(
        "two_tier_career_extension"
    )
    rejected = extension & audited["_audited_extension_season"].ne(True)
    rejected_rows = int(rejected.sum())
    audited = audited[~rejected].drop(
        columns="_audited_extension_season"
    )
    return audited, rejected_rows


def latest_prior_elo(
    side_rows: pd.DataFrame, elo: pd.DataFrame
) -> pd.DataFrame:
    """Attach the latest ClubElo observation strictly before each match date."""
    out = side_rows.copy()
    out["asof_elo"] = np.nan
    out["elo_date"] = pd.NaT
    out["same_day_elo_available"] = False
    elo_groups = {
        int(club_id): group.sort_values("date")
        for club_id, group in elo.groupby("club_id", sort=False)
    }
    for club_id, indices in out.groupby("club_id").groups.items():
        club_id = int(club_id)
        if club_id not in elo_groups:
            continue
        history = elo_groups[club_id]
        history_dates = history["date"].dt.normalize().to_numpy(
            dtype="datetime64[ns]"
        )
        match_dates = out.loc[indices, "date"].dt.normalize().to_numpy(
            dtype="datetime64[ns]"
        )
        inclusive_positions = np.searchsorted(
            history_dates, match_dates, side="right"
        ) - 1
        inclusive_valid = inclusive_positions >= 0
        same_day = np.zeros(len(match_dates), dtype=bool)
        same_day[inclusive_valid] = (
            history_dates[inclusive_positions[inclusive_valid]]
            == match_dates[inclusive_valid]
        )
        out.loc[np.asarray(list(indices)), "same_day_elo_available"] = same_day

        # ``side='left'`` excludes every observation dated on the match day.
        positions = np.searchsorted(
            history_dates, match_dates, side="left"
        ) - 1
        valid = positions >= 0
        valid_indices = np.asarray(list(indices))[valid]
        valid_positions = positions[valid]
        out.loc[valid_indices, "asof_elo"] = history["Elo"].to_numpy()[
            valid_positions
        ]
        out.loc[valid_indices, "elo_date"] = history["date"].to_numpy()[
            valid_positions
        ]
    out["elo_days_from_match"] = (
        out["date"].dt.normalize() - out["elo_date"].dt.normalize()
    ).dt.days
    invalid = out["asof_elo"].notna() & out["elo_days_from_match"].le(0)
    if invalid.any():
        raise RuntimeError(
            f"Strict pre-match Elo audit failed for {int(invalid.sum()):,} sides"
        )
    return out


def build_match_side_elo(
    matches: pd.DataFrame,
    club_seasons: pd.DataFrame,
    reference_leagues: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_sides = matches[[
        "game_id",
        "date",
        "season_start",
        "player_club_id",
        "player_domestic_comp",
    ]].rename(
        columns={
            "player_club_id": "club_id",
            "player_domestic_comp": "domestic_competition_id",
        }
    )
    opponent_sides = matches[[
        "game_id",
        "date",
        "season_start",
        "opponent_id",
        "opponent_domestic_comp",
    ]].rename(
        columns={
            "opponent_id": "club_id",
            "opponent_domestic_comp": "domestic_competition_id",
        }
    )
    sides = pd.concat([player_sides, opponent_sides], ignore_index=True)
    sides = sides.dropna(subset=["club_id", "date"]).drop_duplicates(
        ["game_id", "club_id"]
    )
    sides["club_id"] = sides["club_id"].astype(int)

    needed_clubs = set(sides["club_id"])
    elo = pd.read_csv(
        DATA_DIR / "club_elo_ratings.csv",
        usecols=["transfermarkt_id", "Elo", "date"],
        low_memory=False,
    )
    elo["club_id"] = pd.to_numeric(
        elo["transfermarkt_id"], errors="coerce"
    ).astype("Int64")
    elo = elo[
        elo["club_id"].isin(needed_clubs)
    ].dropna(subset=["club_id", "Elo", "date"])
    elo["club_id"] = elo["club_id"].astype(int)
    elo["date"] = pd.to_datetime(elo["date"], errors="coerce")
    elo = elo.dropna(subset=["date"]).sort_values(["club_id", "date"])

    sides = latest_prior_elo(sides, elo)
    club_season_direct = (
        sides.dropna(subset=["asof_elo", "domestic_competition_id"])
        .groupby(
            [
                "club_id",
                "season_start",
                "domestic_competition_id",
            ],
            as_index=False,
        )
        .agg(club_season_elo=("asof_elo", "median"))
    )
    league_strength = (
        club_season_direct.groupby(
            ["season_start", "domestic_competition_id"],
            as_index=False,
        )
        .agg(
            league_strength_elo=("club_season_elo", "median"),
            direct_clubs=("club_id", "nunique"),
        )
    )
    reference = (
        league_strength[
            league_strength["domestic_competition_id"].isin(
                reference_leagues
            )
        ]
        .groupby("season_start")
        .agg(
            reference_elo=("league_strength_elo", "median"),
            reference_leagues_available=(
                "domestic_competition_id",
                "nunique",
            ),
        )
        .reset_index()
    )
    if reference["reference_leagues_available"].lt(
        len(reference_leagues)
    ).any():
        bad = reference.loc[
            reference["reference_leagues_available"].lt(
                len(reference_leagues)
            )
        ]
        raise RuntimeError(
            "Incomplete Big Five reference seasons: "
            + bad.to_dict(orient="records").__repr__()
        )

    sides = sides.merge(
        league_strength,
        on=["season_start", "domestic_competition_id"],
        how="left",
    ).merge(
        reference[["season_start", "reference_elo"]],
        on="season_start",
        how="left",
        validate="many_to_one",
    )
    sides["elo"] = sides["asof_elo"]
    sides["elo_source"] = np.where(
        sides["asof_elo"].notna(), "asof", ""
    )
    league_fallback = sides["elo"].isna() & sides[
        "league_strength_elo"
    ].notna()
    sides.loc[league_fallback, "elo"] = sides.loc[
        league_fallback, "league_strength_elo"
    ]
    sides.loc[league_fallback, "elo_source"] = "league_season_median"
    reference_fallback = sides["elo"].isna() & sides[
        "reference_elo"
    ].notna()
    sides.loc[reference_fallback, "elo"] = sides.loc[
        reference_fallback, "reference_elo"
    ]
    sides.loc[reference_fallback, "elo_source"] = "top5_reference"
    if sides["elo"].isna().any():
        raise RuntimeError(
            f"Missing final Elo for {int(sides['elo'].isna().sum()):,} "
            "match-club sides"
        )
    return sides, league_strength.merge(
        reference, on="season_start", how="left"
    )


def attach_side_elo(
    matches: pd.DataFrame, sides: pd.DataFrame
) -> pd.DataFrame:
    player = sides.rename(
        columns={
            "club_id": "player_club_id",
            "elo": "player_team_elo",
            "asof_elo": "player_asof_elo",
            "elo_source": "player_elo_source",
            "elo_days_from_match": "player_elo_days_from_match",
            "reference_elo": "top5_reference_elo",
        }
    )[[
        "game_id",
        "player_club_id",
        "player_team_elo",
        "player_asof_elo",
        "player_elo_source",
        "player_elo_days_from_match",
        "top5_reference_elo",
    ]]
    opponent = sides.rename(
        columns={
            "club_id": "opponent_id",
            "elo": "opponent_elo",
            "asof_elo": "opponent_asof_elo",
            "elo_source": "opponent_elo_source",
            "elo_days_from_match": "opponent_elo_days_from_match",
        }
    )[[
        "game_id",
        "opponent_id",
        "opponent_elo",
        "opponent_asof_elo",
        "opponent_elo_source",
        "opponent_elo_days_from_match",
    ]]
    return matches.merge(
        player,
        on=["game_id", "player_club_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        opponent,
        on=["game_id", "opponent_id"],
        how="left",
        validate="many_to_one",
    )


def build_rank_features(
    matches: pd.DataFrame,
    canonical_games: pd.DataFrame,
    club_seasons: pd.DataFrame,
) -> pd.DataFrame:
    club_games = pd.read_csv(
        CANONICAL_DIR / "club_games.csv", low_memory=False
    )
    positions = club_games.dropna(subset=["own_position"]).merge(
        canonical_games[["game_id", "date", "season_start"]],
        on="game_id",
        how="inner",
    )
    positions = positions.sort_values(
        ["club_id", "season_start", "date"]
    )
    positions = (
        positions.groupby(["club_id", "season_start"], as_index=False)
        .tail(1)[["club_id", "season_start", "own_position"]]
        .rename(columns={"own_position": "final_league_position"})
    )

    extension_path = (
        EXTENSION_DIR / "normalized_full" / "club_seasons.parquet"
    )
    extension = pd.read_parquet(
        extension_path,
        columns=["club_id", "season", "final_rank_derived"],
    ).rename(
        columns={
            "season": "season_start",
            "final_rank_derived": "final_league_position",
        }
    )
    extension = extension.dropna(subset=["final_league_position"])
    positions = pd.concat([positions, extension], ignore_index=True)
    positions = positions.drop_duplicates(
        ["club_id", "season_start"], keep="first"
    )

    league_sizes = (
        club_seasons.groupby(
            ["season_start", "domestic_competition_id"]
        )["club_id"]
        .nunique()
        .rename("league_size")
        .reset_index()
    )
    positions = positions.merge(
        club_seasons[[
            "club_id",
            "season_start",
            "domestic_competition_id",
        ]],
        on=["club_id", "season_start"],
        how="left",
    ).merge(
        league_sizes,
        on=["season_start", "domestic_competition_id"],
        how="left",
    )
    positions["club_rank_percentile"] = (
        (positions["final_league_position"] - 1)
        / (positions["league_size"] - 1).replace(0, np.nan)
    ).clip(0, 1)
    return matches.merge(
        positions[[
            "club_id",
            "season_start",
            "final_league_position",
            "league_size",
            "club_rank_percentile",
        ]].rename(columns={"club_id": "player_club_id"}),
        on=["player_club_id", "season_start"],
        how="left",
        validate="many_to_one",
    )


def load_players() -> pd.DataFrame:
    canonical = pd.read_csv(
        CANONICAL_DIR / "players.csv", low_memory=False
    )
    extension_path = (
        EXTENSION_DIR / "normalized_full" / "players.parquet"
    )
    extension = pd.read_parquet(extension_path)
    missing = extension[
        ~extension["player_id"].isin(canonical["player_id"])
    ].copy()
    for column in canonical.columns:
        if column not in missing:
            missing[column] = pd.NA
    players = pd.concat(
        [canonical, missing[canonical.columns]], ignore_index=True
    ).drop_duplicates("player_id")
    for column in extension.columns:
        if column not in players:
            players[column] = players["player_id"].map(
                extension.drop_duplicates("player_id").set_index(
                    "player_id"
                )[column]
            )
    status_path = (
        EXTENSION_DIR / "career_appearances" / "player_fetch_status.csv"
    )
    if status_path.exists():
        status = pd.read_csv(status_path).drop_duplicates(
            "player_id", keep="last"
        )
        status_columns = [
            "first_club_season_api",
            "last_club_season_api",
            "has_pre_window_club_appearance",
            "has_post_window_club_appearance",
        ]
        for column in status_columns:
            if column in status:
                players[column] = players["player_id"].map(
                    status.set_index("player_id")[column]
                )
    return players


def weighted_average(
    frame: pd.DataFrame, value: str, weight: str = "minutes_played"
) -> pd.Series:
    numerator = (
        frame[value].fillna(0) * frame[weight].fillna(0)
    ).groupby([frame["player_id"], frame["season_start"]]).sum()
    denominator = frame[weight].fillna(0).groupby(
        [frame["player_id"], frame["season_start"]]
    ).sum()
    return numerator / denominator.replace(0, np.nan)


def aggregate_player_seasons(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values(
        ["player_id", "season_start", "date", "game_id"]
    )
    group_keys = ["player_id", "season_start"]
    season = matches.groupby(group_keys).agg(
        total_minutes=("minutes_played", "sum"),
        cam_minutes=("cam_minutes", "sum"),
        final_weighted_minutes=("final_weighted_minutes", "sum"),
        games_played=("game_id", "nunique"),
        games_started_90=("minutes_played", lambda x: int((x >= 80).sum())),
        player_club_id=("player_club_id", "last"),
        player_domestic_comp=("player_domestic_comp", "last"),
        clubs_played=("player_club_id", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        final_league_position=("final_league_position", "last"),
        league_size=("league_size", "last"),
    ).reset_index()

    weighted_cols = [
        "player_team_elo",
        "match_quality_elo",
        "match_quality_weight",
        "opponent_elo_gap",
        "opponent_expected_score",
        "opponent_quality_weight",
        "final_minutes_multiplier",
        "club_rank_percentile",
    ]
    for column in weighted_cols:
        values = weighted_average(matches, column).rename(
            {
                "player_team_elo": "club_season_avg_elo",
                "match_quality_elo": "avg_match_quality_elo",
                "match_quality_weight": "avg_match_quality_weight",
                "opponent_elo_gap": "avg_opponent_elo_gap",
                "opponent_expected_score": "avg_opponent_expected_score",
                "opponent_quality_weight": "avg_opponent_quality_weight",
                "final_minutes_multiplier": "avg_final_minutes_multiplier",
                "club_rank_percentile": "club_rank_percentile",
            }[column]
        )
        season = season.merge(
            values.reset_index(), on=group_keys, how="left"
        )

    imputed = (
        matches["player_elo_source"].ne("asof")
        | matches["opponent_elo_source"].ne("asof")
    ).astype(float)
    matches = matches.assign(elo_imputed_match=imputed)
    imputed_share = weighted_average(
        matches, "elo_imputed_match"
    ).rename("elo_imputed_minutes_share")
    season = season.merge(
        imputed_share.reset_index(), on=group_keys, how="left"
    )
    season["data_quality_score"] = (
        1.0 - season["elo_imputed_minutes_share"].fillna(1.0)
    )

    family_minutes = (
        matches.pivot_table(
            index=group_keys,
            columns="competition_family",
            values="minutes_played",
            aggfunc="sum",
            fill_value=0,
        )
        .add_suffix("_minutes")
        .reset_index()
    )
    season = season.merge(family_minutes, on=group_keys, how="left")
    family_columns = [
        column
        for column in season.columns
        if column.endswith("_minutes")
        and column not in {
            "total_minutes",
            "cam_minutes",
            "final_weighted_minutes",
        }
    ]
    for column in family_columns:
        season[column.replace("_minutes", "_minutes_share")] = (
            season[column] / season["total_minutes"].replace(0, np.nan)
        ).fillna(0.0)
    return season


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    start_season = int(config["model_season_start"])
    end_season = int(config["model_season_end"])
    alpha = float(config["opponent_alpha"])
    beta = float(config["quality_beta"])
    gap_scale = float(config["elo_gap_scale"])
    quality_scale = float(config["quality_elo_scale"])
    bounds = tuple(config["final_multiplier_bounds"])
    canonical_start_season = int(config["canonical_start_season"])
    in_scope_domestic_leagues = set(
        config["in_scope_domestic_leagues"]
    )

    print("Loading competition scope...")
    competitions = pd.read_csv(
        CANONICAL_DIR / "competitions.csv", low_memory=False
    )
    scope = build_competition_scope(competitions, config)
    scope.to_csv(OUT_DIR / "competition_scope.csv", index=False)
    canonical_ids = set(competitions["competition_id"])

    print("Loading canonical senior-club appearances...")
    canonical_matches, canonical_games = load_canonical_matches(
        scope, start_season, end_season
    )
    print("Loading audited extension appearances...")
    extension_matches = load_extension_matches(
        scope,
        canonical_ids,
        start_season,
        end_season,
        canonical_start_season,
    )
    shared_columns = sorted(
        set(canonical_matches.columns) | set(extension_matches.columns)
    )
    canonical_matches = canonical_matches.reindex(columns=shared_columns)
    extension_matches = extension_matches.reindex(columns=shared_columns)
    matches = pd.concat(
        [canonical_matches, extension_matches], ignore_index=True
    )
    matches = matches.drop_duplicates("appearance_id", keep="first")
    matches = matches.dropna(
        subset=[
            "date",
            "player_id",
            "player_club_id",
            "opponent_id",
            "minutes_played",
            "competition_family",
        ]
    ).copy()
    matches["player_id"] = matches["player_id"].astype(int)
    matches["player_club_id"] = matches["player_club_id"].astype(int)
    matches["opponent_id"] = matches["opponent_id"].astype(int)
    matches["minutes_played"] = pd.to_numeric(
        matches["minutes_played"], errors="coerce"
    )
    matches = matches[matches["minutes_played"].gt(0)].copy()
    print(
        f"Canonical rows={len(canonical_matches):,}; "
        f"extension rows={len(extension_matches):,}; "
        f"combined unique rows={len(matches):,}"
    )

    print("Building season-specific club competition mapping...")
    club_seasons = build_club_season_competitions(
        matches, in_scope_domestic_leagues
    )
    matches = attach_competition_context(matches, club_seasons)
    before_scope_filter = len(matches)
    matches = matches[
        matches["player_domestic_comp"].isin(in_scope_domestic_leagues)
    ].copy()
    print(
        f"Two-tier club-season scope retained {len(matches):,}/"
        f"{before_scope_filter:,} appearance rows"
    )
    matches, excluded_unaudited_extension_rows = (
        filter_audited_extension_seasons(matches)
    )
    print(
        "Audited extension seasons excluded "
        f"{excluded_unaudited_extension_rows:,} appearance rows"
    )

    print("Attaching latest pre-match ClubElo...")
    sides, league_strength = build_match_side_elo(
        matches, club_seasons, config["reference_leagues"]
    )
    matches = attach_side_elo(matches, sides)
    non_league = ~matches["competition_family"].eq("domestic_league")
    unaudited_non_league = non_league & (
        matches["player_asof_elo"].isna()
        | matches["opponent_asof_elo"].isna()
    )
    excluded_unaudited_non_league_rows = int(unaudited_non_league.sum())
    matches = matches[~unaudited_non_league].copy()
    required_elo = [
        "player_team_elo",
        "opponent_elo",
        "top5_reference_elo",
    ]
    if matches[required_elo].isna().any().any():
        raise RuntimeError(
            "Missing Elo values after audited fallback:\n"
            + matches[required_elo].isna().sum().to_string()
        )

    print("Computing match-quality and opponent weights...")
    matches["match_quality_elo"] = (
        matches["player_team_elo"] + matches["opponent_elo"]
    ) / 2.0
    matches["match_quality_weight"] = 1.0 + beta * np.tanh(
        (
            matches["match_quality_elo"]
            - matches["top5_reference_elo"]
        )
        / quality_scale
    )
    matches["opponent_elo_gap"] = (
        matches["opponent_elo"] - matches["player_team_elo"]
    )
    matches["opponent_expected_score"] = 1.0 / (
        1.0
        + 10.0
        ** (-matches["opponent_elo_gap"] / gap_scale)
    )
    matches["opponent_quality_weight"] = (
        1.0
        + alpha
        * (2.0 * matches["opponent_expected_score"] - 1.0)
    )
    matches["unclipped_minutes_multiplier"] = (
        matches["match_quality_weight"]
        * matches["opponent_quality_weight"]
    )
    matches["final_minutes_multiplier"] = matches[
        "unclipped_minutes_multiplier"
    ].clip(*bounds)
    matches["cam_multiplier"] = matches["final_minutes_multiplier"]
    matches["cam_minutes"] = (
        matches["minutes_played"] * matches["cam_multiplier"]
    )
    matches["final_weighted_minutes"] = matches["cam_minutes"]

    print("Attaching club rank as a feature only...")
    matches = build_rank_features(
        matches, canonical_games, club_seasons
    )

    competition_summary = (
        matches.groupby(
            [
                "season_start",
                "competition_id",
                "competition_family",
            ],
            as_index=False,
        )
        .agg(
            unique_games=("game_id", "nunique"),
            unique_clubs=("player_club_id", "nunique"),
            participant_elo_median=("player_team_elo", "median"),
            match_quality_elo_median=("match_quality_elo", "median"),
            match_quality_weight_median=(
                "match_quality_weight",
                "median",
            ),
            final_multiplier_median=(
                "final_minutes_multiplier",
                "median",
            ),
        )
    )
    competition_summary.to_csv(
        OUT_DIR / "competition_strength_tiers.csv", index=False
    )
    league_strength.to_csv(
        OUT_DIR / "league_season_strength.csv", index=False
    )

    match_path = OUT_DIR / "match_level_weighted_minutes.parquet"
    matches.to_parquet(match_path, index=False)
    print(f"Saved match-level rows: {len(matches):,}")

    print("Aggregating player seasons...")
    player_seasons = aggregate_player_seasons(matches)
    players = load_players()
    players["date_of_birth"] = pd.to_datetime(
        players["date_of_birth"], errors="coerce"
    )
    player_info = players[[
        "player_id",
        "name",
        "position",
        "sub_position",
        "date_of_birth",
        "height_in_cm",
        "foot",
        "first_club_season_api",
        "last_club_season_api",
        "has_pre_window_club_appearance",
        "has_post_window_club_appearance",
    ]]
    player_seasons = player_seasons.merge(
        player_info, on="player_id", how="left", validate="many_to_one"
    )
    season_dates = pd.to_datetime(
        player_seasons["season_start"].astype(str) + "-08-01"
    )
    player_seasons["age_at_season_start"] = (
        season_dates - player_seasons["date_of_birth"]
    ).dt.days / 365.25
    career_bounds = player_seasons.groupby("player_id")["season_start"].agg(
        first_scope_season_observed="min",
        last_scope_season_observed="max",
    )
    player_seasons = player_seasons.merge(
        career_bounds,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    first_age = (
        player_seasons.sort_values("season_start")
        .drop_duplicates("player_id")
        .set_index("player_id")["age_at_season_start"]
    )
    player_seasons["age_at_first_scope_season"] = (
        player_seasons["player_id"].map(first_age)
    )
    exact_pre_window = pd.to_numeric(
        player_seasons.get(
            "has_pre_window_club_appearance",
            pd.Series(0, index=player_seasons.index),
        ),
        errors="coerce",
    ).fillna(0).eq(1)
    boundary_entry = (
        player_seasons["first_scope_season_observed"].eq(start_season)
        & player_seasons["age_at_first_scope_season"].gt(18)
    )
    player_seasons["left_truncated_at_window"] = (
        exact_pre_window | boundary_entry
    ).astype(int)
    api_start_missing = player_seasons.get(
        "first_club_season_api",
        pd.Series(np.nan, index=player_seasons.index),
    ).isna()
    player_seasons["career_history_uncertain"] = (
        player_seasons["left_truncated_at_window"].eq(1)
        | (
            api_start_missing
            & player_seasons["age_at_first_scope_season"].gt(22)
        )
    ).astype(int)
    player_seasons["right_censored_at_window_end"] = (
        player_seasons["last_scope_season_observed"].eq(end_season)
    ).astype(int)
    player_seasons.to_parquet(
        OUT_DIR / "player_season_weighted_minutes.parquet", index=False
    )

    family_counts = (
        matches.groupby("competition_family")["game_id"]
        .nunique()
        .sort_values(ascending=False)
        .astype(int)
        .to_dict()
    )
    checks = {
        "model_season_start": start_season,
        "model_season_end": end_season,
        "canonical_match_rows": int(len(canonical_matches)),
        "audited_extension_match_rows": int(len(extension_matches)),
        "combined_match_rows": int(len(matches)),
        "two_tier_domestic_leagues": sorted(in_scope_domestic_leagues),
        "excluded_unaudited_non_league_rows": (
            excluded_unaudited_non_league_rows
        ),
        "excluded_unaudited_extension_rows": (
            excluded_unaudited_extension_rows
        ),
        "unique_games": int(matches["game_id"].nunique()),
        "unique_players": int(matches["player_id"].nunique()),
        "player_season_rows": int(len(player_seasons)),
        "competition_families": family_counts,
        "asof_player_elo_share": float(
            matches["player_elo_source"].eq("asof").mean()
        ),
        "asof_opponent_elo_share": float(
            matches["opponent_elo_source"].eq("asof").mean()
        ),
        "strict_previous_day_elo": True,
        "same_day_elo_available_sides": int(
            sides["same_day_elo_available"].sum()
        ),
        "same_day_elo_available_share": float(
            sides["same_day_elo_available"].mean()
        ),
        "minimum_direct_elo_days_before_match": int(
            sides.loc[
                sides["asof_elo"].notna(), "elo_days_from_match"
            ].min()
        ),
        "quality_weight_min": float(
            matches["match_quality_weight"].min()
        ),
        "quality_weight_max": float(
            matches["match_quality_weight"].max()
        ),
        "opponent_weight_min": float(
            matches["opponent_quality_weight"].min()
        ),
        "opponent_weight_max": float(
            matches["opponent_quality_weight"].max()
        ),
        "final_multiplier_min": float(
            matches["final_minutes_multiplier"].min()
        ),
        "final_multiplier_max": float(
            matches["final_minutes_multiplier"].max()
        ),
        "final_multiplier_clipped_share": float((
            ~np.isclose(
                matches["unclipped_minutes_multiplier"],
                matches["final_minutes_multiplier"],
            )
        ).mean()),
        "club_rank_used_in_minutes": False,
        "national_team_matches_included": False,
        "unclassified_extension_competitions_included": False,
        "left_truncated_players": int(
            player_seasons.loc[
                player_seasons["left_truncated_at_window"].eq(1),
                "player_id",
            ].nunique()
        ),
        "career_history_uncertain_players": int(
            player_seasons.loc[
                player_seasons["career_history_uncertain"].eq(1),
                "player_id",
            ].nunique()
        ),
        "right_censored_players": int(
            player_seasons.loc[
                player_seasons["right_censored_at_window_end"].eq(1),
                "player_id",
            ].nunique()
        ),
    }
    (REPORT_DIR / "match_quality_weighting_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))
    print(
        f"Saved player-season rows: {len(player_seasons):,} to "
        f"{OUT_DIR / 'player_season_weighted_minutes.parquet'}"
    )


if __name__ == "__main__":
    main()
