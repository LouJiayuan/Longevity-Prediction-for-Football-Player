#!/usr/bin/env python3
"""Align a long Transfermarkt-derived injury history to the modelling panel.

The source contains direct Transfermarkt player IDs, but absence of a row does
not prove absence of injury. This builder therefore keeps reported values
nullable and creates zero-filled variants only under explicitly named coverage
assumptions. Injury spells are split across July-June seasons, and implausibly
long spells are retained for audit but excluded from validated days-lost
features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
# data/ 默认与代码目录同级；不存在时回退到上一级目录。
DATA_ROOT = BASE_DIR / "data" if (BASE_DIR / "data").is_dir() else PROJECT_DIR / "data"
DEFAULT_PANEL = DATA_ROOT / "processed" / "player_season_features.parquet"
DEFAULT_OUT_DIR = DATA_ROOT / "injuries" / "longitudinal_2000_2025"

DATASET_HANDLE = "xfkzujqjvx97n/football-datasets"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_HANDLE}"
DATASET_REPOSITORY = "https://github.com/salimt/football-datasets"
DATASET_LICENSE = "CC0: Public Domain (Kaggle data card)"
INJURY_RELATIVE_PATH = Path("player_injuries/player_injuries.csv")
PROFILE_RELATIVE_PATH = Path("player_profiles/player_profiles.csv")

MIN_SEASON = 2000
MAX_SEASON = 2025
MAX_VALIDATED_SPELL_DAYS = 730

ILLNESS_PATTERN = re.compile(
    r"ill|flu|virus|corona|covid|infection|fever|cold|quarant|disease|"
    r"food poisoning|tonsill|gastro|stomach bug",
    flags=re.IGNORECASE,
)
REST_FITNESS_PATTERN = re.compile(r"\brest\b|fitness", flags=re.IGNORECASE)
UNKNOWN_PATTERN = re.compile(r"^\s*(unknown injury|unknown|not reported|n/a)?\s*$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Directory containing player_injuries/ and player_profiles/.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-season", type=int, default=MIN_SEASON)
    parser.add_argument("--max-season", type=int, default=MAX_SEASON)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_dataset_root() -> Path | None:
    versions = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / DATASET_HANDLE
        / "versions"
    )
    if not versions.exists():
        return None
    candidates = [
        path
        for path in versions.iterdir()
        if (path / INJURY_RELATIVE_PATH).exists()
        and (path / PROFILE_RELATIVE_PATH).exists()
    ]
    if not candidates:
        return None

    def version_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.name), path.name
        except ValueError:
            return -1, path.name

    return max(candidates, key=version_key)


def resolve_source_root(explicit: Path | None, download: bool) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
    else:
        root = cached_dataset_root()
        if root is None and download:
            try:
                import kagglehub
            except ImportError as exc:
                raise RuntimeError("--download requires kagglehub") from exc
            root = Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()
        if root is None:
            raise FileNotFoundError(
                f"Dataset {DATASET_HANDLE} is not cached. Use --download or --source-root."
            )
    for relative_path in (INJURY_RELATIVE_PATH, PROFILE_RELATIVE_PATH):
        if not (root / relative_path).exists():
            raise FileNotFoundError(root / relative_path)
    return root


def season_from_date(value: pd.Timestamp) -> int | None:
    if pd.isna(value):
        return None
    return int(value.year if value.month >= 7 else value.year - 1)


def season_from_label(value: object) -> int | None:
    match = re.match(r"\s*(\d{2,4})\s*/", str(value))
    if not match:
        return None
    year = int(match.group(1))
    if len(match.group(1)) == 2:
        year += 2000 if year <= 30 else 1900
    return year


def absence_category(reason: object) -> str:
    value = "" if pd.isna(reason) else str(reason).strip()
    if UNKNOWN_PATTERN.search(value):
        return "unknown"
    if ILLNESS_PATTERN.search(value):
        return "illness"
    if REST_FITNESS_PATTERN.search(value):
        return "rest_or_fitness"
    return "injury_or_other"


def load_panel(path: Path) -> pd.DataFrame:
    columns = ["player_id", "season_start", "name", "date_of_birth"]
    panel = pd.read_parquet(path, columns=columns)
    panel["player_id"] = pd.to_numeric(panel["player_id"], errors="coerce").astype("Int64")
    panel["season_start"] = pd.to_numeric(
        panel["season_start"], errors="coerce"
    ).astype("Int64")
    panel["date_of_birth"] = pd.to_datetime(panel["date_of_birth"], errors="coerce")
    panel = panel.dropna(subset=["player_id", "season_start"]).copy()
    if panel.duplicated(["player_id", "season_start"]).any():
        raise ValueError("The modelling panel has duplicate player-season keys")
    dob_variants = panel.groupby("player_id")["date_of_birth"].nunique(dropna=True)
    if dob_variants.gt(1).any():
        raise ValueError("The modelling panel has conflicting birth dates for a player ID")
    return panel


def load_profiles(path: Path) -> pd.DataFrame:
    profiles = pd.read_csv(
        path,
        usecols=["player_id", "player_name", "date_of_birth"],
        low_memory=False,
    )
    profiles["player_id"] = pd.to_numeric(
        profiles["player_id"], errors="coerce"
    ).astype("Int64")
    profiles["source_date_of_birth"] = pd.to_datetime(
        profiles["date_of_birth"], errors="coerce"
    )
    profiles = profiles.dropna(subset=["player_id"]).copy()
    if profiles.duplicated("player_id").any():
        profiles = profiles.drop_duplicates("player_id", keep="first")
    profiles["source_profile_present"] = True
    return profiles[[
        "player_id", "player_name", "source_date_of_birth", "source_profile_present"
    ]]


def prepare_events(
    source: pd.DataFrame,
    profiles: pd.DataFrame,
    panel: pd.DataFrame,
    min_season: int,
    max_season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "player_id",
        "season_name",
        "injury_reason",
        "from_date",
        "end_date",
        "days_missed",
        "games_missed",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Longitudinal injury source is missing: {sorted(missing)}")

    events = source.copy()
    raw_rows = len(events)
    events["source_row"] = np.arange(len(events), dtype=int)
    events["player_id"] = pd.to_numeric(events["player_id"], errors="coerce").astype("Int64")
    events["injury_start_date"] = pd.to_datetime(events["from_date"], errors="coerce")
    events["reported_end_date"] = pd.to_datetime(events["end_date"], errors="coerce")
    events["reported_days_missed"] = pd.to_numeric(
        events["days_missed"], errors="coerce"
    )
    events["reported_games_missed"] = pd.to_numeric(
        events["games_missed"], errors="coerce"
    )
    events["injury_reason_key"] = (
        events["injury_reason"].fillna("").astype(str).str.strip().str.lower()
    )

    duplicate_keys = [
        "player_id",
        "injury_start_date",
        "reported_end_date",
        "injury_reason_key",
        "reported_days_missed",
        "reported_games_missed",
    ]
    events["source_duplicate_rows"] = events.groupby(
        duplicate_keys, dropna=False
    )["source_row"].transform("size")
    events = events.drop_duplicates(duplicate_keys, keep="first").copy()
    deduplicated_rows = len(events)

    valid_reported_end = (
        events["injury_start_date"].notna()
        & events["reported_end_date"].notna()
        & events["reported_end_date"].ge(events["injury_start_date"])
    )
    inferred_end = events["injury_start_date"] + pd.to_timedelta(
        events["reported_days_missed"].sub(1).clip(lower=0), unit="D"
    )
    can_infer_end = (
        events["injury_start_date"].notna()
        & events["reported_days_missed"].ge(1)
    )
    events["resolved_end_date"] = events["reported_end_date"].where(valid_reported_end)
    events.loc[
        events["resolved_end_date"].isna() & can_infer_end, "resolved_end_date"
    ] = inferred_end
    events["end_date_inferred"] = ~valid_reported_end & can_infer_end
    events["invalid_reported_date_order"] = (
        events["injury_start_date"].notna()
        & events["reported_end_date"].notna()
        & events["reported_end_date"].lt(events["injury_start_date"])
    )
    events["resolved_calendar_days"] = (
        events["resolved_end_date"] - events["injury_start_date"]
    ).dt.days + 1
    events["duration_disagreement_days"] = (
        events["reported_days_missed"] - events["resolved_calendar_days"]
    ).abs()

    events["source_season_start"] = events["season_name"].map(season_from_label).astype("Int64")
    events["date_season_start"] = events["injury_start_date"].map(season_from_date).astype("Int64")
    events["source_date_season_mismatch"] = (
        events["source_season_start"].notna()
        & events["date_season_start"].notna()
        & events["source_season_start"].ne(events["date_season_start"])
    )
    events["absence_category"] = events["injury_reason"].map(absence_category)

    panel_identity = panel[["player_id", "date_of_birth"]].drop_duplicates("player_id")
    panel_identity = panel_identity.rename(columns={"date_of_birth": "panel_date_of_birth"})
    panel_identity["panel_player_present"] = True
    events = events.merge(profiles, on="player_id", how="left", validate="many_to_one")
    events = events.merge(panel_identity, on="player_id", how="left", validate="many_to_one")
    events["source_profile_present"] = (
        events["source_profile_present"].astype("boolean").fillna(False).astype(bool)
    )
    events["panel_player_present"] = (
        events["panel_player_present"].astype("boolean").fillna(False).astype(bool)
    )
    both_dob = events["source_date_of_birth"].notna() & events["panel_date_of_birth"].notna()
    dob_delta_days = (
        events["source_date_of_birth"] - events["panel_date_of_birth"]
    ).dt.days.abs()
    events["source_profile_dob_exact_match"] = (
        both_dob & events["source_date_of_birth"].eq(events["panel_date_of_birth"])
    )
    events["source_profile_dob_within_one_day"] = both_dob & dob_delta_days.le(1)
    events["source_profile_dob_conflict"] = (
        both_dob & dob_delta_days.gt(1)
    )
    events["identity_accepted"] = (
        events["panel_player_present"] & ~events["source_profile_dob_conflict"]
    )

    events["long_spell_anomaly"] = (
        events["reported_days_missed"].gt(MAX_VALIDATED_SPELL_DAYS)
        | events["resolved_calendar_days"].gt(MAX_VALIDATED_SPELL_DAYS)
    )
    events["duration_validated"] = (
        events["injury_start_date"].notna()
        & events["resolved_end_date"].notna()
        & events["resolved_calendar_days"].between(1, MAX_VALIDATED_SPELL_DAYS)
        & ~events["long_spell_anomaly"]
    )
    events["event_validated"] = events["identity_accepted"] & events["duration_validated"]

    source_starts_in_window = events["date_season_start"].between(min_season, max_season)
    deduplicated_rows_starting_in_window = int(source_starts_in_window.sum())
    deduplicated_players_starting_in_window = int(
        events.loc[source_starts_in_window, "player_id"].nunique()
    )
    deduplicated_start_date_min = str(events["injury_start_date"].min().date())
    deduplicated_start_date_max = str(events["injury_start_date"].max().date())

    window_start = pd.Timestamp(year=min_season, month=7, day=1)
    window_end = pd.Timestamp(year=max_season + 1, month=6, day=30)
    overlaps_window = (
        events["injury_start_date"].le(window_end)
        & events["resolved_end_date"].ge(window_start)
    )
    events = events[events["panel_player_present"] & overlaps_window].copy()
    events = events.sort_values(["player_id", "injury_start_date", "source_row"])
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=int)

    checks = {
        "raw_source_rows": int(raw_rows),
        "raw_source_players": int(source["player_id"].nunique()),
        "rows_after_exact_deduplication": int(deduplicated_rows),
        "deduplicated_rows_starting_in_window": deduplicated_rows_starting_in_window,
        "deduplicated_players_starting_in_window": deduplicated_players_starting_in_window,
        "deduplicated_start_date_min": deduplicated_start_date_min,
        "deduplicated_start_date_max": deduplicated_start_date_max,
        "panel_window_event_rows": int(len(events)),
        "panel_window_players": int(events["player_id"].nunique()),
        "validated_event_rows": int(events["event_validated"].sum()),
        "long_spell_anomaly_rows": int(events["long_spell_anomaly"].sum()),
        "dob_conflict_event_rows": int(events["source_profile_dob_conflict"].sum()),
        "source_date_season_mismatch_rows": int(events["source_date_season_mismatch"].sum()),
    }
    return events, checks


def split_events_across_seasons(
    events: pd.DataFrame,
    min_season: int,
    max_season: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        start = event.injury_start_date
        end = event.resolved_end_date
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        first_season = max(min_season, season_from_date(start))
        last_season = min(max_season, season_from_date(end))
        if first_season > last_season:
            continue
        event_start_season = season_from_date(start)
        for season_start in range(first_season, last_season + 1):
            season_date_start = pd.Timestamp(year=season_start, month=7, day=1)
            season_date_end = pd.Timestamp(year=season_start + 1, month=6, day=30)
            overlap_start = max(start, season_date_start)
            overlap_end = min(end, season_date_end)
            overlap_days = int((overlap_end - overlap_start).days + 1)
            started_here = event_start_season == season_start
            identity_accepted = bool(event.identity_accepted)
            validated = bool(event.event_validated)
            category = event.absence_category
            duration = event.resolved_calendar_days
            rows.append({
                "event_id": event.event_id,
                "player_id": event.player_id,
                "season_start": season_start,
                "overlap_start_date": overlap_start,
                "overlap_end_date": overlap_end,
                "overlap_days_raw": overlap_days,
                "overlap_days_validated": float(overlap_days) if validated else np.nan,
                "identity_accepted": identity_accepted,
                "event_validated": validated,
                "event_started_this_season": started_here,
                "accepted_event_start": int(started_here and identity_accepted),
                "validated_event_start": int(started_here and validated),
                "severe_event_start": int(started_here and validated and duration >= 28),
                "major_event_start": int(started_here and validated and duration >= 90),
                "long_spell_anomaly_start": int(
                    started_here and identity_accepted and bool(event.long_spell_anomaly)
                ),
                "injury_or_other_start": int(
                    started_here and identity_accepted and category == "injury_or_other"
                ),
                "illness_start": int(
                    started_here and identity_accepted and category == "illness"
                ),
                "rest_or_fitness_start": int(
                    started_here and identity_accepted and category == "rest_or_fitness"
                ),
                "unknown_reason_start": int(
                    started_here and identity_accepted and category == "unknown"
                ),
                "games_missed_at_start": (
                    event.reported_games_missed
                    if started_here and identity_accepted
                    else np.nan
                ),
            })
    segments = pd.DataFrame(rows)
    if segments.empty:
        raise ValueError("No injury spell overlaps the requested season window")
    segments["player_id"] = segments["player_id"].astype("Int64")
    segments["season_start"] = segments["season_start"].astype("Int64")
    return segments


def unique_covered_days(group: pd.DataFrame) -> int:
    intervals = sorted(
        zip(group["overlap_start_date"], group["overlap_end_date"]),
        key=lambda pair: pair[0],
    )
    merged: list[list[pd.Timestamp]] = []
    for start, end in intervals:
        separated = (
            not merged
            or (pd.Timestamp(start) - pd.Timestamp(merged[-1][1])).days > 1
        )
        if separated:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return int(sum((end - start).days + 1 for start, end in merged))


def aggregate_segments(segments: pd.DataFrame) -> pd.DataFrame:
    keys = ["player_id", "season_start"]
    accepted = segments[segments["identity_accepted"]].copy()
    grouped = (
        accepted.groupby(keys, as_index=False)
        .agg(
            reported_active_spell_count=("event_id", "nunique"),
            reported_injury_start_count=("accepted_event_start", "sum"),
            validated_injury_start_count=("validated_event_start", "sum"),
            severe_injury_start_count=("severe_event_start", "sum"),
            major_injury_start_count=("major_event_start", "sum"),
            long_spell_anomaly_start_count=("long_spell_anomaly_start", "sum"),
            injury_or_other_start_count=("injury_or_other_start", "sum"),
            illness_start_count=("illness_start", "sum"),
            rest_or_fitness_start_count=("rest_or_fitness_start", "sum"),
            unknown_reason_start_count=("unknown_reason_start", "sum"),
            reported_games_missed_at_start=("games_missed_at_start", "sum"),
        )
    )
    validated = segments[segments["event_validated"]].copy()
    validated_counts = (
        validated.groupby(keys, as_index=False)
        .agg(
            validated_active_spell_count=("event_id", "nunique"),
            validated_injury_days_sum=("overlap_days_validated", "sum"),
        )
    )
    union_days = (
        validated.groupby(keys, sort=False)
        .apply(unique_covered_days, include_groups=False)
        .rename("validated_unique_injury_days")
        .reset_index()
    )
    grouped = grouped.merge(validated_counts, on=keys, how="left", validate="one_to_one")
    grouped = grouped.merge(union_days, on=keys, how="left", validate="one_to_one")
    return grouped.sort_values(keys).reset_index(drop=True)


def reporting_era(season: pd.Series) -> pd.Series:
    conditions = [
        season.le(2006),
        season.between(2007, 2015),
        season.between(2016, 2024),
        season.eq(2025),
    ]
    labels = ["early_sparse", "partial", "main_window", "incomplete_2025"]
    return pd.Series(np.select(conditions, labels, default="outside_scope"), index=season.index)


def align_to_panel(
    panel: pd.DataFrame,
    profiles: pd.DataFrame,
    all_source_events: pd.DataFrame,
    season_features: pd.DataFrame,
) -> pd.DataFrame:
    aligned = panel.copy()
    if "source_profile_present" not in profiles:
        profiles = profiles.copy()
        profiles["source_profile_present"] = True
    profile_map = profiles[[
        "player_id", "source_date_of_birth", "source_profile_present"
    ]].copy()
    aligned = aligned.merge(profile_map, on="player_id", how="left", validate="many_to_one")
    aligned["source_profile_present"] = (
        aligned["source_profile_present"].astype("boolean").fillna(False).astype(bool)
    )
    both_dob = aligned["source_date_of_birth"].notna() & aligned["date_of_birth"].notna()
    dob_delta_days = (
        aligned["source_date_of_birth"] - aligned["date_of_birth"]
    ).dt.days.abs()
    aligned["source_profile_dob_exact_match"] = (
        both_dob & aligned["source_date_of_birth"].eq(aligned["date_of_birth"])
    )
    aligned["source_profile_dob_within_one_day"] = both_dob & dob_delta_days.le(1)
    aligned["source_profile_dob_conflict"] = (
        both_dob & dob_delta_days.gt(1)
    )
    history_ids = set(
        pd.to_numeric(all_source_events["player_id"], errors="coerce").dropna().astype(int)
    )
    aligned["source_player_has_any_injury_history"] = aligned["player_id"].isin(history_ids)
    aligned = aligned.merge(
        season_features,
        on=["player_id", "season_start"],
        how="left",
        validate="one_to_one",
    )

    observed_columns = [
        "reported_active_spell_count",
        "reported_injury_start_count",
        "validated_injury_start_count",
        "validated_active_spell_count",
        "validated_injury_days_sum",
        "validated_unique_injury_days",
        "severe_injury_start_count",
        "major_injury_start_count",
        "long_spell_anomaly_start_count",
        "injury_or_other_start_count",
        "illness_start_count",
        "rest_or_fitness_start_count",
        "unknown_reason_start_count",
        "reported_games_missed_at_start",
    ]
    aligned["injury_event_recorded_this_season"] = (
        aligned["reported_injury_start_count"].fillna(0).gt(0)
    )
    aligned["injury_active_spell_recorded_this_season"] = (
        aligned["reported_active_spell_count"].fillna(0).gt(0)
    )
    aligned["injury_reporting_era"] = reporting_era(aligned["season_start"])
    aligned["injury_main_window"] = aligned["season_start"].between(2016, 2024)
    aligned["injury_sensitivity_window"] = aligned["season_start"].between(2007, 2015)
    aligned["injury_descriptive_only_window"] = (
        aligned["season_start"].between(2000, 2006) | aligned["season_start"].eq(2025)
    )
    aligned["injury_record_status"] = np.select(
        [
            aligned["source_profile_dob_conflict"],
            aligned["injury_active_spell_recorded_this_season"],
            aligned["source_player_has_any_injury_history"],
            aligned["source_profile_present"],
        ],
        [
            "id_dob_conflict",
            "reported_event_or_active_spell",
            "history_player_no_record_this_season",
            "profile_present_no_record_this_season",
        ],
        default="player_not_in_source_profiles",
    )
    aligned["missing_event_proves_zero_injury"] = False

    profile_coverage = (
        aligned["source_profile_present"] & ~aligned["source_profile_dob_conflict"]
    )
    history_coverage = (
        aligned["source_player_has_any_injury_history"]
        & ~aligned["source_profile_dob_conflict"]
    )
    for column in observed_columns:
        aligned[f"{column}_profile_zero_assumption"] = aligned[column].fillna(0).where(
            profile_coverage
        )
        aligned[f"{column}_history_zero_assumption"] = aligned[column].fillna(0).where(
            history_coverage
        )
    return aligned


def coverage_by_season(aligned: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        aligned.groupby("season_start", as_index=False)
        .agg(
            panel_player_seasons=("player_id", "size"),
            panel_players=("player_id", "nunique"),
            source_profile_player_seasons=("source_profile_present", "sum"),
            history_player_seasons=("source_player_has_any_injury_history", "sum"),
            event_start_player_seasons=("injury_event_recorded_this_season", "sum"),
            active_spell_player_seasons=("injury_active_spell_recorded_this_season", "sum"),
            reported_injury_starts=("reported_injury_start_count", "sum"),
            validated_injury_starts=("validated_injury_start_count", "sum"),
            validated_unique_injury_days=("validated_unique_injury_days", "sum"),
            long_spell_anomaly_starts=("long_spell_anomaly_start_count", "sum"),
        )
    )
    for numerator in [
        "source_profile_player_seasons",
        "history_player_seasons",
        "event_start_player_seasons",
        "active_spell_player_seasons",
    ]:
        coverage[f"{numerator}_share"] = (
            coverage[numerator] / coverage["panel_player_seasons"]
        )
    coverage["reporting_era"] = reporting_era(coverage["season_start"])
    return coverage


def validate_outputs(
    events: pd.DataFrame,
    segments: pd.DataFrame,
    aligned: pd.DataFrame,
    min_season: int,
    max_season: int,
) -> None:
    if aligned.duplicated(["player_id", "season_start"]).any():
        raise ValueError("Aligned injury panel has duplicate player-seasons")
    if not aligned["season_start"].between(min_season, max_season).all():
        raise ValueError("Aligned injury panel contains seasons outside the requested window")
    if segments["overlap_days_raw"].le(0).any():
        raise ValueError("Spell splitting produced non-positive overlap days")
    if aligned["validated_unique_injury_days"].dropna().gt(366).any():
        raise ValueError("Unique validated injury days exceed a season length")
    anomalous_ids = set(events.loc[events["long_spell_anomaly"], "event_id"])
    anomaly_segments = segments[segments["event_id"].isin(anomalous_ids)]
    if anomaly_segments["overlap_days_validated"].notna().any():
        raise ValueError("An anomalously long spell entered validated days")
    no_record = ~aligned["injury_active_spell_recorded_this_season"]
    if aligned.loc[no_record, "reported_active_spell_count"].notna().any():
        raise ValueError("Missing event seasons were silently encoded as observed zeros")


def main() -> None:
    args = parse_args()
    source_root = resolve_source_root(args.source_root, args.download)
    injury_path = source_root / INJURY_RELATIVE_PATH
    profile_path = source_root / PROFILE_RELATIVE_PATH
    panel = load_panel(args.panel)
    panel = panel[panel["season_start"].between(args.min_season, args.max_season)].copy()
    profiles = load_profiles(profile_path)
    source_events = pd.read_csv(injury_path, low_memory=False)

    events, event_checks = prepare_events(
        source_events,
        profiles,
        panel,
        args.min_season,
        args.max_season,
    )
    segments = split_events_across_seasons(events, args.min_season, args.max_season)
    season_features = aggregate_segments(segments)
    aligned = align_to_panel(panel, profiles, source_events, season_features)
    coverage = coverage_by_season(aligned)
    validate_outputs(events, segments, aligned, args.min_season, args.max_season)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out_dir / "injury_events_aligned.parquet", index=False)
    segments.to_parquet(args.out_dir / "injury_spell_season_segments.parquet", index=False)
    aligned.to_parquet(args.out_dir / "player_season_injury_features.parquet", index=False)
    coverage.to_csv(args.out_dir / "injury_coverage_by_season.csv", index=False)
    events[events["long_spell_anomaly"]].to_csv(
        args.out_dir / "injury_long_spell_audit.csv", index=False
    )
    events[events["source_profile_dob_conflict"]].to_csv(
        args.out_dir / "injury_identity_conflict_audit.csv", index=False
    )

    manifest = {
        "purpose": "Longitudinal public injury/availability proxy aligned to the modelling panel",
        "source": {
            "dataset": DATASET_HANDLE,
            "url": DATASET_URL,
            "repository": DATASET_REPOSITORY,
            "license": DATASET_LICENSE,
            "source_root": str(source_root),
            "injury_file_sha256": sha256_file(injury_path),
            "profile_file_sha256": sha256_file(profile_path),
        },
        "panel": {
            "path": str(args.panel.resolve()),
            "season_window": [args.min_season, args.max_season],
            "player_seasons": int(len(aligned)),
            "players": int(aligned["player_id"].nunique()),
        },
        "checks": {
            **event_checks,
            "aligned_player_seasons": int(len(aligned)),
            "aligned_players": int(aligned["player_id"].nunique()),
            "reported_event_start_player_seasons": int(
                aligned["injury_event_recorded_this_season"].sum()
            ),
            "players_with_any_source_injury_history": int(
                aligned.loc[
                    aligned["source_player_has_any_injury_history"], "player_id"
                ].nunique()
            ),
            "source_profile_players_in_panel": int(
                aligned.loc[aligned["source_profile_present"], "player_id"].nunique()
            ),
            "source_profile_dob_conflict_players": int(
                aligned.loc[aligned["source_profile_dob_conflict"], "player_id"].nunique()
            ),
        },
        "measurement_rules": {
            "season_definition": "July 1 through June 30",
            "spells_crossing_seasons": "Split by calendar-day overlap",
            "overlapping_spells": "Validated unique days use interval union, bounded by season length",
            "maximum_validated_spell_days": MAX_VALIDATED_SPELL_DAYS,
            "identity_date_tolerance": (
                "Direct Transfermarkt player ID plus date of birth within one day; "
                "the one-day tolerance addresses a systematic legacy date-import shift"
            ),
            "missing_event_semantics": "Unknown, never automatically observed zero",
            "zero_variants": [
                "profile_zero_assumption",
                "history_zero_assumption",
            ],
        },
        "recommended_windows": {
            "main_injury_ablation": "2016-2024",
            "coverage_sensitivity": "2007-2015",
            "descriptive_only": "2000-2006 and 2025",
        },
        "model_use_status": (
            "Aligned but not activated in the final model. First run a separate temporal "
            "injury-feature ablation in 2016-2024 and retain missingness indicators; do not "
            "treat no reported event as verified health."
        ),
        "limitations": [
            "Transfermarkt injury histories are crowd-sourced availability proxies, not clinical records.",
            "Historical reporting intensity changes sharply across seasons.",
            "Games missed are assigned to the event-start season because match-level allocation is unavailable.",
            "Very long unresolved spells are retained for audit and excluded from validated days.",
        ],
    }
    (args.out_dir / "injury_alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["checks"], indent=2))
    print(f"Wrote longitudinal injury alignment to {args.out_dir}")


if __name__ == "__main__":
    main()
