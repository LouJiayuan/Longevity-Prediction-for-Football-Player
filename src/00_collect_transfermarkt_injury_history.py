#!/usr/bin/env python3
"""Collect selected Transfermarkt player injury pages with resume support.

This is a research collector, not a bulk mirroring tool. Use a conservative
delay, collect only the required cohort, and review Transfermarkt's current
terms before collection or redistribution. The normalized output can be passed
to ``00_build_public_injury_data.py --transfermarkt-events``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
# data/ 默认与代码目录同级；不存在时回退到上一级目录。
DATA_ROOT = BASE_DIR / "data" if (BASE_DIR / "data").is_dir() else PROJECT_DIR / "data"
DEFAULT_PLAYERS = DATA_ROOT / "transfermarkt_public" / "players.csv"
DEFAULT_OUT_DIR = DATA_ROOT / "injuries" / "transfermarkt_collected"
USER_AGENT = "football-longevity-dissertation/1.0 (academic research)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    parser.add_argument("--player-id", type=int, action="append", default=[])
    parser.add_argument(
        "--player-ids-file",
        type=Path,
        help="CSV containing player_id, or a text file containing one ID per line.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--max-players", type=int)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def requested_ids(args: argparse.Namespace) -> list[int]:
    values = list(args.player_id)
    if args.player_ids_file:
        if args.player_ids_file.suffix.lower() == ".csv":
            frame = pd.read_csv(args.player_ids_file)
            if "player_id" not in frame:
                raise ValueError("--player-ids-file CSV must contain player_id")
            values.extend(pd.to_numeric(frame["player_id"], errors="coerce").dropna().astype(int))
        else:
            values.extend(
                int(line.strip())
                for line in args.player_ids_file.read_text().splitlines()
                if line.strip()
            )
    if not values:
        raise ValueError("Provide --player-id or --player-ids-file")
    ordered = list(dict.fromkeys(values))
    return ordered[: args.max_players] if args.max_players else ordered


def injury_url(profile_url: str, player_id: int) -> str:
    match = re.search(r"transfermarkt\.[^/]+/([^/]+)/", str(profile_url))
    player_code = match.group(1) if match else f"player-{player_id}"
    return (
        f"https://www.transfermarkt.co.uk/{player_code}/verletzungen/"
        f"spieler/{player_id}/plus/1"
    )


def parse_days(value: str) -> float:
    match = re.search(r"\d+", value.replace(",", ""))
    return float(match.group()) if match else float("nan")


def season_start(value: str, start_date: pd.Timestamp) -> int | None:
    match = re.match(r"\s*(\d{2,4})\s*/", value)
    if match:
        year = int(match.group(1))
        return 2000 + year if year < 100 else year
    if pd.isna(start_date):
        return None
    return int(start_date.year if start_date.month >= 7 else start_date.year - 1)


def parse_injury_page(html: str, player_id: int, player_name: str, url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = None
    for candidate in soup.select("table.items"):
        headers = [cell.get_text(" ", strip=True).lower() for cell in candidate.select("th")]
        if {"season", "injury", "from", "until", "days"}.issubset(headers):
            table = candidate
            break
    if table is None:
        raise ValueError("No injury-history table found")

    rows = []
    for row in table.select("tbody tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
        if len(cells) < 6:
            continue
        season, injury, start_text, end_text, days_text, games_text = cells[:6]
        start = pd.to_datetime(start_text, format="%d/%m/%Y", errors="coerce")
        end = pd.to_datetime(end_text, format="%d/%m/%Y", errors="coerce")
        rows.append({
            "player_id": player_id,
            "tm_player_name": player_name,
            "season_start": season_start(season, start),
            "injury_start_date": start,
            "injury_end_date": end,
            "days_lost": parse_days(days_text),
            "games_missed": pd.to_numeric(games_text.replace("-", ""), errors="coerce"),
            "injury_type": injury,
            "source_url": url,
        })
    return pd.DataFrame(rows)


def combine_cached(cache_dir: Path, out_path: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(cache_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(out_path, index=False)
    return combined


def main() -> None:
    args = parse_args()
    ids = requested_ids(args)
    players = pd.read_csv(
        args.players,
        usecols=["player_id", "name", "url"],
        low_memory=False,
    )
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    players = players[players["player_id"].isin(ids)].drop_duplicates("player_id")
    missing_ids = sorted(set(ids) - set(players["player_id"].dropna().astype(int)))
    if missing_ids:
        raise ValueError(f"Player IDs absent from players.csv: {missing_ids[:20]}")

    cache_dir = args.out_dir / "players"
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "collection_status.csv"
    status_rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.9"})

    for index, player in enumerate(players.itertuples(index=False), start=1):
        player_id = int(player.player_id)
        cache_path = cache_dir / f"{player_id}.csv"
        url = injury_url(player.url, player_id)
        if cache_path.exists() and not args.force:
            status_rows.append({
                "player_id": player_id,
                "player_name": player.name,
                "status": "cached",
                "source_url": url,
            })
            continue
        if index > 1:
            time.sleep(max(args.sleep_seconds, 0))
        try:
            response = session.get(url, timeout=args.timeout)
            response.raise_for_status()
            frame = parse_injury_page(response.text, player_id, player.name, url)
            frame.to_csv(cache_path, index=False)
            status = "collected" if not frame.empty else "collected_no_rows"
            error = ""
        except Exception as exc:  # Preserve progress and make failures auditable.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        status_rows.append({
            "player_id": player_id,
            "player_name": player.name,
            "status": status,
            "error": error,
            "source_url": url,
        })
        pd.DataFrame(status_rows).to_csv(status_path, index=False)

    status = pd.DataFrame(status_rows)
    status.to_csv(status_path, index=False)
    combined = combine_cached(cache_dir, args.out_dir / "transfermarkt_injury_events.csv")
    manifest = {
        "requested_players": len(ids),
        "status_counts": status["status"].value_counts().to_dict(),
        "event_rows": int(len(combined)),
        "terms_note": (
            "Transfermarkt website terms apply. This collector does not grant "
            "permission to redistribute scraped content."
        ),
    }
    (args.out_dir / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
