"""Diagnose information loss when raw minutes replace adjusted minutes.

The official 2000-2025 label is based on adjusted minutes. This script does
not change that label. It recreates the same strict-prior-peak, 65%, two
adjacent-season rule using raw minutes, then measures agreement in the period
where both quantities are observed.

Outputs:
- raw_vs_adjusted_overlap_summary.csv
- raw_vs_adjusted_overlap_by_league.csv
- raw_vs_adjusted_overlap_player_events.csv
- raw_vs_adjusted_overlap_report.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


KIMI_DIR = Path(__file__).resolve().parents[1]
# data/ 在工作区位于 kimi/ 上一层，在公开快照中与代码同级；两种布局都能解析。
DATA_ROOT = KIMI_DIR / "data" if (KIMI_DIR / "data").is_dir() else KIMI_DIR.parent / "data"
DATA_PATH = DATA_ROOT / "processed" / "player_season_labeled.parquet"
OUT_DIR = KIMI_DIR / "outputs"

PEAK_RATIO = 0.65
MIN_HISTORY_SEASONS = 2
REQUIRED_COLUMNS = {
    "player_id",
    "season_start",
    "player_domestic_comp",
    "total_minutes",
    "final_weighted_minutes",
    "dropoff_candidate",
    "dropoff_confirmed",
    "dropoff_event",
    "dropoff_onset",
}


def sustained_flags(
    df: pd.DataFrame, minutes_column: str
) -> pd.DataFrame:
    """Build labels using only values available at or before each season."""
    out = df[["player_id", "season_start", minutes_column]].copy()
    out = out.sort_values(["player_id", "season_start"]).reset_index()
    out = out.rename(columns={"index": "source_index"})

    n_rows = len(out)
    history_count = np.zeros(n_rows, dtype=int)
    peak_minutes = np.full(n_rows, np.nan)
    peak_season = np.full(n_rows, np.nan)

    for indices in out.groupby("player_id", sort=False).groups.values():
        positions = np.asarray(list(indices), dtype=int)
        minutes = out.loc[positions, minutes_column].to_numpy(dtype=float)
        seasons = out.loc[positions, "season_start"].to_numpy(dtype=float)
        best_position: int | None = None

        for position, row_position in enumerate(positions):
            history_count[row_position] = position
            if best_position is not None:
                peak_minutes[row_position] = minutes[best_position]
                peak_season[row_position] = seasons[best_position]
            # Retain the most recent season on a tied peak, matching script 02.
            if best_position is None or minutes[position] >= minutes[best_position]:
                best_position = position

    out["history_seasons_before"] = history_count
    out["prior_peak_minutes"] = peak_minutes
    out["prior_peak_season"] = peak_season
    out["minutes_vs_prior_peak"] = (
        out[minutes_column] / out["prior_peak_minutes"].replace(0, np.nan)
    )
    eligible = (
        out["history_seasons_before"].ge(MIN_HISTORY_SEASONS)
        & out["prior_peak_minutes"].gt(0)
    )
    low = eligible & out["minutes_vs_prior_peak"].lt(PEAK_RATIO)
    prior_low = low.groupby(out["player_id"]).shift(1, fill_value=False)
    prior_season = out.groupby("player_id")["season_start"].shift(1)
    confirmed = low & prior_low & out["season_start"].eq(prior_season + 1)
    prior_confirmation = confirmed.groupby(out["player_id"]).transform(
        lambda values: values.shift(1, fill_value=False).cummax()
    )
    event = confirmed & ~prior_confirmation
    onset = event.groupby(out["player_id"]).shift(-1, fill_value=False)

    out["candidate"] = low.astype(int)
    out["confirmed"] = confirmed.astype(int)
    out["event"] = event.astype(int)
    out["onset"] = onset.astype(int)
    return out.set_index("source_index").sort_index()


def binary_agreement(
    reference: pd.Series, comparison: pd.Series
) -> dict[str, float | int]:
    """Treat the adjusted label as reference and raw label as comparison."""
    reference = reference.astype(int)
    comparison = comparison.astype(int)
    true_positive = int(((reference == 1) & (comparison == 1)).sum())
    false_negative = int(((reference == 1) & (comparison == 0)).sum())
    false_positive = int(((reference == 0) & (comparison == 1)).sum())
    true_negative = int(((reference == 0) & (comparison == 0)).sum())
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return {
        "rows": int(len(reference)),
        "adjusted_positive_rows": int(reference.sum()),
        "raw_positive_rows": int(comparison.sum()),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "agreement": float((reference == comparison).mean()),
        "cohen_kappa": float(cohen_kappa_score(reference, comparison)),
        "raw_precision_vs_adjusted": float(precision),
        "raw_recall_vs_adjusted": float(recall),
    }


def first_event_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one event season per player for either adjusted or raw labels."""
    adjusted = (
        df.loc[df["dropoff_event"].eq(1), [
            "player_id", "season_start", "player_domestic_comp"
        ]]
        .sort_values(["player_id", "season_start"])
        .drop_duplicates("player_id")
        .rename(columns={
            "season_start": "adjusted_event_season",
            "player_domestic_comp": "adjusted_event_league",
        })
    )
    raw = (
        df.loc[df["raw_event"].eq(1), [
            "player_id", "season_start", "player_domestic_comp"
        ]]
        .sort_values(["player_id", "season_start"])
        .drop_duplicates("player_id")
        .rename(columns={
            "season_start": "raw_event_season",
            "player_domestic_comp": "raw_event_league",
        })
    )
    events = adjusted.merge(raw, on="player_id", how="outer")
    events["event_season_difference_raw_minus_adjusted"] = (
        events["raw_event_season"] - events["adjusted_event_season"]
    )
    events["comparison_league"] = events[
        "adjusted_event_league"
    ].fillna(events["raw_event_league"])
    events["event_comparison"] = np.select(
        [
            events["adjusted_event_season"].notna()
            & events["raw_event_season"].notna()
            & events["event_season_difference_raw_minus_adjusted"].eq(0),
            events["adjusted_event_season"].notna()
            & events["raw_event_season"].notna()
            & events["event_season_difference_raw_minus_adjusted"].lt(0),
            events["adjusted_event_season"].notna()
            & events["raw_event_season"].notna(),
            events["adjusted_event_season"].notna(),
        ],
        ["same_season", "raw_earlier", "raw_later", "adjusted_only"],
        default="raw_only",
    )
    return events.sort_values("player_id").reset_index(drop=True)


def league_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate candidate and first-event agreement by current league."""
    rows: list[dict[str, float | int | str]] = []
    for league, group in df.groupby("player_domestic_comp", dropna=False):
        for label in ("candidate", "event"):
            metrics = binary_agreement(
                group[f"dropoff_{label}"], group[f"raw_{label}"]
            )
            rows.append({
                "player_domestic_comp": league if pd.notna(league) else "missing",
                "label": label,
                **metrics,
            })
    return pd.DataFrame(rows).sort_values(["label", "player_domestic_comp"])


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values(["player_id", "season_start"]).reset_index(drop=True)

    adjusted = sustained_flags(df, "final_weighted_minutes")
    raw = sustained_flags(df, "total_minutes")
    for label in ("candidate", "confirmed", "event", "onset"):
        official = df[f"dropoff_{label}"].astype(int)
        rebuilt = adjusted[label].astype(int)
        if not official.equals(rebuilt):
            mismatches = int(official.ne(rebuilt).sum())
            raise RuntimeError(
                f"Adjusted-label recreation failed for {label}: {mismatches} rows."
            )

    df["raw_prior_peak_minutes"] = raw["prior_peak_minutes"]
    df["raw_prior_peak_season"] = raw["prior_peak_season"]
    df["raw_minutes_vs_prior_peak"] = raw["minutes_vs_prior_peak"]
    for label in ("candidate", "confirmed", "event", "onset"):
        df[f"raw_{label}"] = raw[label].astype(int)

    ratio = df["final_weighted_minutes"] / df["total_minutes"]
    raw_adjusted_correlation = df["total_minutes"].corr(
        df["final_weighted_minutes"]
    )
    candidate_metrics = binary_agreement(
        df["dropoff_candidate"], df["raw_candidate"]
    )
    event_metrics = binary_agreement(df["dropoff_event"], df["raw_event"])
    onset_metrics = binary_agreement(df["dropoff_onset"], df["raw_onset"])
    events = first_event_table(df)
    both_events = events.dropna(
        subset=["adjusted_event_season", "raw_event_season"]
    ).copy()
    timing = both_events["event_season_difference_raw_minus_adjusted"]

    summary = pd.DataFrame([
        {
            "metric_group": "data",
            "metric": "player_season_rows",
            "value": int(len(df)),
        },
        {
            "metric_group": "data",
            "metric": "unique_players",
            "value": int(df["player_id"].nunique()),
        },
        {
            "metric_group": "data",
            "metric": "adjusted_to_raw_multiplier_median",
            "value": float(ratio.median()),
        },
        {
            "metric_group": "data",
            "metric": "adjusted_to_raw_multiplier_p05",
            "value": float(ratio.quantile(0.05)),
        },
        {
            "metric_group": "data",
            "metric": "adjusted_to_raw_multiplier_p95",
            "value": float(ratio.quantile(0.95)),
        },
        {
            "metric_group": "data",
            "metric": "raw_adjusted_minutes_pearson_correlation",
            "value": float(raw_adjusted_correlation),
        },
    ])
    for name, metrics in {
        "candidate": candidate_metrics,
        "event": event_metrics,
        "onset": onset_metrics,
    }.items():
        summary = pd.concat([
            summary,
            pd.DataFrame([
                {"metric_group": name, "metric": key, "value": value}
                for key, value in metrics.items()
            ]),
        ], ignore_index=True)
    event_counts = events["event_comparison"].value_counts()
    for name in [
        "same_season", "raw_earlier", "raw_later", "adjusted_only", "raw_only"
    ]:
        summary.loc[len(summary)] = {
            "metric_group": "first_event_timing",
            "metric": f"players_{name}",
            "value": int(event_counts.get(name, 0)),
        }
    for statistic, value in {
        "both_event_players": len(both_events),
        "median_raw_minus_adjusted_seasons": timing.median(),
        "mean_absolute_event_season_difference": timing.abs().mean(),
        "same_season_share_among_both": timing.eq(0).mean(),
        "within_one_season_share_among_both": timing.abs().le(1).mean(),
    }.items():
        summary.loc[len(summary)] = {
            "metric_group": "first_event_timing",
            "metric": statistic,
            "value": float(value),
        }

    league = league_summary(df)
    summary_path = OUT_DIR / "raw_vs_adjusted_overlap_summary.csv"
    league_path = OUT_DIR / "raw_vs_adjusted_overlap_by_league.csv"
    events_path = OUT_DIR / "raw_vs_adjusted_overlap_player_events.csv"
    report_path = OUT_DIR / "raw_vs_adjusted_overlap_report.md"
    summary.to_csv(summary_path, index=False)
    league.to_csv(league_path, index=False)
    events.to_csv(events_path, index=False)

    report = f"""# Raw vs Adjusted Minutes: 2000-2025 Overlap Diagnostic

## Design

The official adjusted-minute label was recreated from scratch and matched the
stored candidate, confirmation, event and onset flags exactly. The comparison
then applied the **same** strict-prior-peak, 65% and two-adjacent-season rule
to `total_minutes` only. The raw label is an analytic counterfactual; it does
not replace the official outcome.

## Scale of the adjustment

- Player-season rows: {len(df):,}
- Players: {df['player_id'].nunique():,}
- Correlation, raw vs adjusted minutes: {raw_adjusted_correlation:.3f}
- Adjusted/raw multiplier: median {ratio.median():.3f}; 5th-95th percentile
  {ratio.quantile(0.05):.3f}-{ratio.quantile(0.95):.3f}.

The high raw/adjusted correlation measures retained volume information. The
multiplier range measures the context information that raw minutes discard.

## Label agreement

| Label | Adjusted positive rows | Raw positive rows | Agreement | Cohen's kappa | Raw precision vs adjusted | Raw recall vs adjusted |
|---|---:|---:|---:|---:|---:|---:|
| Candidate | {candidate_metrics['adjusted_positive_rows']:,} | {candidate_metrics['raw_positive_rows']:,} | {format_percent(candidate_metrics['agreement'])} | {candidate_metrics['cohen_kappa']:.3f} | {format_percent(candidate_metrics['raw_precision_vs_adjusted'])} | {format_percent(candidate_metrics['raw_recall_vs_adjusted'])} |
| First event | {event_metrics['adjusted_positive_rows']:,} | {event_metrics['raw_positive_rows']:,} | {format_percent(event_metrics['agreement'])} | {event_metrics['cohen_kappa']:.3f} | {format_percent(event_metrics['raw_precision_vs_adjusted'])} | {format_percent(event_metrics['raw_recall_vs_adjusted'])} |
| Onset | {onset_metrics['adjusted_positive_rows']:,} | {onset_metrics['raw_positive_rows']:,} | {format_percent(onset_metrics['agreement'])} | {onset_metrics['cohen_kappa']:.3f} | {format_percent(onset_metrics['raw_precision_vs_adjusted'])} | {format_percent(onset_metrics['raw_recall_vs_adjusted'])} |

## First-event timing

- Players with both a raw and adjusted event: {len(both_events):,}
- Same event season: {int(event_counts.get('same_season', 0)):,}
  ({format_percent(timing.eq(0).mean())} of players with both events)
- Raw event earlier: {int(event_counts.get('raw_earlier', 0)):,}
- Raw event later: {int(event_counts.get('raw_later', 0)):,}
- Adjusted-only event: {int(event_counts.get('adjusted_only', 0)):,}
- Raw-only event: {int(event_counts.get('raw_only', 0)):,}
- Median raw minus adjusted event season: {timing.median():.1f}
- Mean absolute event-season difference: {timing.abs().mean():.2f}
- Within one season: {format_percent(timing.abs().le(1).mean())}

## Interpretation

Raw minutes preserve the amount a player played, but cannot retain the
match-quality and opponent-difficulty adjustments. Therefore raw-only labels
should not be pooled with the adjusted-minute primary outcome. The per-league
table and player event table identify where a separate historical raw-minute
cohort may still be useful as exploratory evidence.

## Files

- `{summary_path.name}`: machine-readable overall metrics.
- `{league_path.name}`: candidate and first-event agreement by league.
- `{events_path.name}`: one record per player with raw/adjusted first-event
  seasons and timing category.
"""
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved: {summary_path}")
    print(f"Saved: {league_path}")
    print(f"Saved: {events_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
