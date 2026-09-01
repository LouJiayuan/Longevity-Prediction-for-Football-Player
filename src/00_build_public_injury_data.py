#!/usr/bin/env python3
"""Build auditable public injury tables without inventing injury events.

Two Kaggle sources are supported:

1. European Football Injuries (2020-2025), an event-level table.
2. Football Player Injury Data (2016-2021), a player-season aggregate built
   from FIFA attributes and Transfermarkt injury histories.

The sources have different units and are therefore retained separately before
being combined into a source-aware player-season table. The script never
expands aggregate days into synthetic events and never overwrites the legacy
``data/injuries/injuries.csv`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


KIMI_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = KIMI_DIR.parent
# data/ 在工作区位于 kimi/ 上一层，在公开快照中与代码同级；两种布局都能解析。
DATA_ROOT = KIMI_DIR / "data" if (KIMI_DIR / "data").is_dir() else PROJECT_DIR / "data"
INJURY_DIR = DATA_ROOT / "injuries"
DEFAULT_OUT_DIR = INJURY_DIR / "public_processed"
DEFAULT_PLAYERS = DATA_ROOT / "transfermarkt_public" / "players.csv"

EVENT_HANDLE = "sananmuzaffarov/european-football-injuries-2020-2025"
EVENT_FILENAME = "full_dataset_thesis - 1.csv"
EVENT_URL = f"https://www.kaggle.com/datasets/{EVENT_HANDLE}"
EVENT_LICENSE = "CC BY-SA 4.0"

AGGREGATE_HANDLE = "kolambekalpesh/football-player-injury-data"
AGGREGATE_FILENAME = "dataset.csv"
AGGREGATE_URL = f"https://www.kaggle.com/datasets/{AGGREGATE_HANDLE}"
AGGREGATE_LICENSE = "Unknown on Kaggle; verify before redistribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    parser.add_argument("--event-source", type=Path)
    parser.add_argument("--aggregate-source", type=Path)
    parser.add_argument(
        "--transfermarkt-events",
        type=Path,
        help="Optional normalized output from 00_collect_transfermarkt_injury_history.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing Kaggle sources with kagglehub.",
    )
    parser.add_argument(
        "--include-medium-confidence",
        action="store_true",
        help="Include unique-name mappings with an age mismatch in canonical events.",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    ascii_value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]", "", ascii_value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_kaggle_file(handle: str, filename: str) -> Path | None:
    versions = Path.home() / ".cache" / "kagglehub" / "datasets" / handle / "versions"
    if not versions.exists():
        return None
    candidates = list(versions.glob(f"*/{filename}"))
    if not candidates:
        return None

    def version_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.parent.name), path.parent.name
        except ValueError:
            return -1, path.parent.name

    return max(candidates, key=version_key)


def resolve_kaggle_source(
    explicit_path: Path | None,
    handle: str,
    filename: str,
    download: bool,
) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(explicit_path)
        return explicit_path.resolve()

    cached = cached_kaggle_file(handle, filename)
    if cached is not None:
        return cached.resolve()
    if not download:
        raise FileNotFoundError(
            f"Missing Kaggle source {handle}. Re-run with --download or provide its CSV path."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("--download requires kagglehub") from exc
    root = Path(kagglehub.dataset_download(handle))
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Downloaded {handle}, but {filename} was not present")
    return path.resolve()


def load_transfermarkt_players(path: Path) -> pd.DataFrame:
    required = ["player_id", "name", "date_of_birth", "url"]
    players = pd.read_csv(path, usecols=lambda column: column in required, low_memory=False)
    missing = {"player_id", "name", "date_of_birth"} - set(players.columns)
    if missing:
        raise ValueError(f"Transfermarkt players file is missing: {sorted(missing)}")
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    players["tm_player_name"] = players["name"].astype(str)
    players["name_key"] = players["name"].map(normalize_name)
    players["tm_date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    return players.dropna(subset=["player_id"]).copy()


def age_on_date(date: pd.Timestamp, birth_date: pd.Timestamp) -> float:
    if pd.isna(date) or pd.isna(birth_date):
        return np.nan
    return float(np.floor((date - birth_date).days / 365.2425))


def parse_days(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def season_start_from_label(value: object, fallback_date: pd.Timestamp) -> float:
    match = re.match(r"\s*(\d{2,4})\s*/", str(value))
    if match:
        year = int(match.group(1))
        return float(2000 + year if year < 100 else year)
    if pd.isna(fallback_date):
        return np.nan
    return float(fallback_date.year if fallback_date.month >= 7 else fallback_date.year - 1)


def map_event_source(source: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Season", "Injury", "Days", "Games missed", "injury_from_parsed",
        "injury_until_parsed", "player_name", "player_age", "club", "league",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Event injury source is missing: {sorted(missing)}")

    candidates = {
        key: group.to_dict("records")
        for key, group in players.groupby("name_key", sort=False)
        if key
    }
    rows: list[dict[str, object]] = []
    for source_row, row in source.reset_index(drop=True).iterrows():
        start = pd.to_datetime(row["injury_from_parsed"], errors="coerce")
        end = pd.to_datetime(row["injury_until_parsed"], errors="coerce")
        source_age = pd.to_numeric(pd.Series([row["player_age"]]), errors="coerce").iloc[0]
        name_key = normalize_name(row["player_name"])
        player_candidates = candidates.get(name_key, [])

        selected: dict[str, object] | None = None
        mapping_method = "unmatched_name"
        mapping_confidence = "unresolved"
        age_difference = np.nan

        if len(player_candidates) == 1:
            selected = player_candidates[0]
            calculated_age = age_on_date(start, selected["tm_date_of_birth"])
            age_difference = abs(calculated_age - source_age) if pd.notna(source_age) else np.nan
            if pd.notna(age_difference) and age_difference <= 1:
                mapping_method = "unique_normalized_name_and_age"
                mapping_confidence = "high"
            else:
                mapping_method = "unique_normalized_name_age_mismatch"
                mapping_confidence = "medium"
        elif len(player_candidates) > 1:
            age_matches = []
            for candidate in player_candidates:
                calculated_age = age_on_date(start, candidate["tm_date_of_birth"])
                if pd.notna(source_age) and pd.notna(calculated_age) and abs(calculated_age - source_age) <= 1:
                    age_matches.append((candidate, abs(calculated_age - source_age)))
            if len(age_matches) == 1:
                selected, age_difference = age_matches[0]
                mapping_method = "ambiguous_name_resolved_by_age"
                mapping_confidence = "high"
            elif len(age_matches) > 1:
                mapping_method = "ambiguous_name_age_tie"
            else:
                mapping_method = "ambiguous_name_no_age_match"

        days_lost = parse_days(row["Days"])
        calculated_days = (end - start).days + 1 if pd.notna(start) and pd.notna(end) else np.nan
        if pd.isna(days_lost):
            days_lost = calculated_days

        rows.append({
            "source_dataset": "kaggle_european_football_injuries_2020_2025",
            "source_row": source_row,
            "source_player_key": name_key,
            "source_player_name": row["player_name"],
            "source_player_age": source_age,
            "player_id": selected["player_id"] if selected else pd.NA,
            "tm_player_name": selected["tm_player_name"] if selected else pd.NA,
            "tm_date_of_birth": selected["tm_date_of_birth"] if selected else pd.NaT,
            "mapping_method": mapping_method,
            "mapping_confidence": mapping_confidence,
            "mapping_age_difference": age_difference,
            "injury_start_date": start,
            "injury_end_date": end,
            "season_start": season_start_from_label(row["Season"], start),
            "days_lost": days_lost,
            "calculated_calendar_days": calculated_days,
            "games_missed": pd.to_numeric(
                pd.Series([row["Games missed"]]), errors="coerce"
            ).iloc[0],
            "injury_type": row["Injury"],
            "club": row["club"],
            "league": row["league"],
            "source_url": EVENT_URL,
            "source_license": EVENT_LICENSE,
        })

    mapped = pd.DataFrame(rows)
    mapped["player_id"] = mapped["player_id"].astype("Int64")
    mapped["season_start"] = mapped["season_start"].astype("Int64")
    mapped["manual_review_required"] = mapped["mapping_confidence"].ne("high")
    return mapped


def map_aggregate_source(source: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    required = {"p_id2", "dob", "start_year", "season_days_injured"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Aggregate injury source is missing: {sorted(missing)}")

    work = source.copy()
    work["source_player_key"] = work["p_id2"].map(normalize_name)
    work["source_date_of_birth"] = pd.to_datetime(work["dob"], errors="coerce")
    work["season_start"] = pd.to_numeric(work["start_year"], errors="coerce").astype("Int64")
    work["season_days_injured"] = pd.to_numeric(
        work["season_days_injured"], errors="coerce"
    )

    identity_map = work[["source_player_key", "source_date_of_birth"]].drop_duplicates()
    candidate = identity_map.merge(
        players[["player_id", "tm_player_name", "tm_date_of_birth", "name_key"]],
        left_on="source_player_key",
        right_on="name_key",
        how="left",
    )
    candidate = candidate[candidate["source_date_of_birth"].eq(candidate["tm_date_of_birth"])]
    counts = candidate.groupby(
        ["source_player_key", "source_date_of_birth"], dropna=False
    )["player_id"].nunique()
    unique_keys = counts[counts.eq(1)].index
    unique_map = (
        candidate.set_index(["source_player_key", "source_date_of_birth"])
        .loc[unique_keys]
        .reset_index()
        .drop_duplicates(["source_player_key", "source_date_of_birth"])
    )
    work = work.merge(
        unique_map[[
            "source_player_key", "source_date_of_birth", "player_id",
            "tm_player_name", "tm_date_of_birth",
        ]],
        on=["source_player_key", "source_date_of_birth"],
        how="left",
    )

    identity_candidate_count = counts.rename("identity_candidate_count").reset_index()
    work = work.merge(
        identity_candidate_count,
        on=["source_player_key", "source_date_of_birth"],
        how="left",
    )
    work["identity_candidate_count"] = work["identity_candidate_count"].fillna(0).astype(int)
    work["mapping_method"] = np.where(
        work["player_id"].notna(),
        "normalized_name_and_exact_birth_date",
        "unresolved_name_birth_date",
    )
    work["mapping_confidence"] = np.where(work["player_id"].notna(), "high", "unresolved")

    group_keys = ["source_player_key", "source_date_of_birth", "season_start"]
    duplicate_stats = (
        work.groupby(group_keys, dropna=False)
        .agg(
            source_duplicate_rows=("p_id2", "size"),
            source_days_variants=("season_days_injured", "nunique"),
        )
        .reset_index()
    )
    work = work.merge(duplicate_stats, on=group_keys, how="left")
    work["source_conflict"] = work["source_days_variants"].gt(1)

    # Duplicate FIFA-position rows often repeat the same injury value. Retain
    # one source identity-season row; conflicting values remain audit-only.
    work = (
        work.sort_values(group_keys)
        .drop_duplicates(group_keys, keep="first")
        .reset_index(drop=True)
    )
    work["manual_review_required"] = (
        work["mapping_confidence"].ne("high") | work["source_conflict"]
    )
    work["source_dataset"] = "kaggle_football_player_injury_data_2016_2021"
    work["source_url"] = AGGREGATE_URL
    work["source_license"] = AGGREGATE_LICENSE
    work["player_id"] = work["player_id"].astype("Int64")

    keep = [
        "source_dataset", "p_id2", "source_player_key", "source_date_of_birth",
        "player_id", "tm_player_name", "tm_date_of_birth", "mapping_method",
        "mapping_confidence", "identity_candidate_count", "manual_review_required",
        "season_start", "season_days_injured", "total_days_injured",
        "source_duplicate_rows", "source_days_variants", "source_conflict",
        "source_url", "source_license",
    ]
    return work[[column for column in keep if column in work.columns]]


def normalize_direct_transfermarkt_events(path: Path) -> pd.DataFrame:
    direct = pd.read_csv(path, low_memory=False)
    required = {"player_id", "injury_start_date", "injury_end_date", "injury_type"}
    missing = required - set(direct.columns)
    if missing:
        raise ValueError(f"Direct Transfermarkt injury file is missing: {sorted(missing)}")
    direct["player_id"] = pd.to_numeric(direct["player_id"], errors="coerce").astype("Int64")
    direct["injury_start_date"] = pd.to_datetime(direct["injury_start_date"], errors="coerce")
    direct["injury_end_date"] = pd.to_datetime(direct["injury_end_date"], errors="coerce")
    if "days_lost" not in direct:
        direct["days_lost"] = (
            direct["injury_end_date"] - direct["injury_start_date"]
        ).dt.days + 1
    direct["days_lost"] = pd.to_numeric(direct["days_lost"], errors="coerce")
    if "season_start" not in direct:
        direct["season_start"] = direct["injury_start_date"].map(
            lambda date: season_start_from_label("", date)
        )
    direct["season_start"] = pd.to_numeric(direct["season_start"], errors="coerce").astype("Int64")
    direct["source_dataset"] = "transfermarkt_player_injury_history_direct"
    direct["mapping_method"] = "transfermarkt_player_id_from_page"
    direct["mapping_confidence"] = "high"
    direct["manual_review_required"] = False
    if "source_url" not in direct:
        direct["source_url"] = "https://www.transfermarkt.co.uk/"
    direct["source_license"] = "Website terms apply; verify before redistribution"
    for column in ["games_missed", "club", "league", "tm_player_name"]:
        if column not in direct:
            direct[column] = pd.NA
    return direct


def canonical_event_table(
    event_mappings: pd.DataFrame,
    include_medium: bool,
    direct_path: Path | None,
) -> pd.DataFrame:
    allowed = ["high", "medium"] if include_medium else ["high"]
    events = event_mappings[event_mappings["mapping_confidence"].isin(allowed)].copy()
    if direct_path is not None:
        direct = normalize_direct_transfermarkt_events(direct_path)
        common = sorted(set(events.columns) | set(direct.columns))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries",
                category=FutureWarning,
            )
            events = pd.concat(
                [direct.reindex(columns=common), events.reindex(columns=common)],
                ignore_index=True,
            )

    events["injury_type_key"] = events["injury_type"].map(normalize_name)
    events["source_priority"] = events["source_dataset"].map({
        "transfermarkt_player_injury_history_direct": 0,
        "kaggle_european_football_injuries_2020_2025": 1,
    }).fillna(9)
    dedupe_keys = [
        "player_id", "injury_start_date", "injury_end_date", "injury_type_key"
    ]
    events = (
        events.sort_values(dedupe_keys + ["source_priority"])
        .drop_duplicates(dedupe_keys, keep="first")
        .drop(columns=["injury_type_key", "source_priority"])
        .reset_index(drop=True)
    )
    events["player_id"] = events["player_id"].astype("Int64")
    return events


def build_source_aware_seasons(
    events: pd.DataFrame,
    aggregate_mappings: pd.DataFrame,
) -> pd.DataFrame:
    event_seasons = (
        events.dropna(subset=["player_id", "season_start"])
        .groupby(["player_id", "season_start"], as_index=False)
        .agg(
            event_injury_count=("injury_start_date", "count"),
            event_days_lost=("days_lost", "sum"),
            event_games_missed=("games_missed", "sum"),
            event_source_count=("source_dataset", "nunique"),
        )
    )
    aggregate_seasons = aggregate_mappings[
        aggregate_mappings["mapping_confidence"].eq("high")
        & ~aggregate_mappings["source_conflict"]
    ][["player_id", "season_start", "season_days_injured"]].rename(
        columns={"season_days_injured": "aggregate_days_lost"}
    )
    combined = event_seasons.merge(
        aggregate_seasons,
        on=["player_id", "season_start"],
        how="outer",
        validate="one_to_one",
    )
    combined["cross_source_overlap"] = (
        combined["event_days_lost"].notna() & combined["aggregate_days_lost"].notna()
    )
    combined["cross_source_days_difference"] = (
        combined["event_days_lost"] - combined["aggregate_days_lost"]
    )
    combined["preferred_days_lost"] = combined["event_days_lost"].combine_first(
        combined["aggregate_days_lost"]
    )
    combined["preferred_source"] = np.where(
        combined["event_days_lost"].notna(),
        "event_history",
        "season_aggregate",
    )
    combined["measurement_zero_is_observed"] = False
    return combined.sort_values(["player_id", "season_start"]).reset_index(drop=True)


def validate_outputs(events: pd.DataFrame, seasons: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        raise ValueError("No canonical injury events were produced")
    if events["player_id"].isna().any():
        raise ValueError("Canonical injury events contain missing player IDs")
    if (events["days_lost"].dropna() < 0).any():
        raise ValueError("Canonical injury events contain negative days lost")
    if (events["injury_end_date"] < events["injury_start_date"]).fillna(False).any():
        raise ValueError("Canonical injury events contain end dates before start dates")
    if seasons.duplicated(["player_id", "season_start"]).any():
        raise ValueError("Source-aware season output has duplicate player-seasons")
    return {
        "canonical_event_rows": int(len(events)),
        "canonical_event_players": int(events["player_id"].nunique()),
        "canonical_event_date_min": str(events["injury_start_date"].min().date()),
        "canonical_event_date_max": str(events["injury_start_date"].max().date()),
        "source_aware_season_rows": int(len(seasons)),
        "source_aware_season_players": int(seasons["player_id"].nunique()),
        "cross_source_overlap_rows": int(seasons["cross_source_overlap"].sum()),
    }


def main() -> None:
    args = parse_args()
    event_path = resolve_kaggle_source(
        args.event_source, EVENT_HANDLE, EVENT_FILENAME, args.download
    )
    aggregate_path = resolve_kaggle_source(
        args.aggregate_source, AGGREGATE_HANDLE, AGGREGATE_FILENAME, args.download
    )
    players = load_transfermarkt_players(args.players)

    event_mappings = map_event_source(pd.read_csv(event_path), players)
    aggregate_mappings = map_aggregate_source(pd.read_csv(aggregate_path), players)
    events = canonical_event_table(
        event_mappings,
        include_medium=args.include_medium_confidence,
        direct_path=args.transfermarkt_events,
    )
    seasons = build_source_aware_seasons(events, aggregate_mappings)
    checks = validate_outputs(events, seasons)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_mappings.to_csv(args.out_dir / "event_mapping_audit.csv", index=False)
    aggregate_mappings.to_csv(args.out_dir / "aggregate_mapping_audit.csv", index=False)
    events.to_csv(args.out_dir / "injury_events_public.csv", index=False)
    seasons.to_csv(args.out_dir / "injury_seasons_source_aware.csv", index=False)

    manifest = {
        "purpose": "Auditable public injury-source reconstruction",
        "legacy_injuries_csv_used": False,
        "medium_confidence_included": bool(args.include_medium_confidence),
        "sources": [
            {
                "dataset": EVENT_HANDLE,
                "url": EVENT_URL,
                "license": EVENT_LICENSE,
                "local_path": str(event_path),
                "sha256": sha256_file(event_path),
                "rows": int(len(event_mappings)),
                "mapping_confidence": event_mappings["mapping_confidence"].value_counts().to_dict(),
            },
            {
                "dataset": AGGREGATE_HANDLE,
                "url": AGGREGATE_URL,
                "license": AGGREGATE_LICENSE,
                "local_path": str(aggregate_path),
                "sha256": sha256_file(aggregate_path),
                "rows_after_identity_season_deduplication": int(len(aggregate_mappings)),
                "mapping_confidence": aggregate_mappings["mapping_confidence"].value_counts().to_dict(),
            },
        ],
        "optional_transfermarkt_events": str(args.transfermarkt_events) if args.transfermarkt_events else None,
        "checks": checks,
        "model_use_status": (
            "Not activated. Validate source coverage, reporting bias and the unknown "
            "license of the aggregate Kaggle dataset before model inclusion."
        ),
    }
    (args.out_dir / "injury_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["checks"], indent=2))
    print(f"Wrote public injury reconstruction to {args.out_dir}")


if __name__ == "__main__":
    main()
