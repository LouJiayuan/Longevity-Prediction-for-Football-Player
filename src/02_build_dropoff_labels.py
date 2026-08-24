"""
Construct sustained drop-off labels from player-season weighted minutes.

The primary event is confirmed only after two consecutive calendar seasons in
which context-adjusted minutes remain below 65% of the player's own prior
career peak. The first low season is retained as onset and the second as the
confirmation/event season. Alternative thresholds are reported only as a
sensitivity analysis.

Outputs:
- player_season_labeled.parquet
- player_survival_events.parquet
- dropoff_threshold_sensitivity.csv
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = DATA_DIR / 'processed'
REPORT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_PEAK_RATIO = 0.65
MIN_HISTORY_SEASONS = 2
RECOVERY_RATIO = 0.80


def season_start_from_date(value):
    if pd.isna(value):
        return np.nan
    return value.year if value.month >= 8 else value.year - 1


def add_rolling_peak(df, col, out_prefix):
    """Retain lagged rolling summaries used by downstream feature engineering."""
    out = df.copy()
    grouped = out.groupby('player_id')[col]
    out[f'{out_prefix}_peak_3yr'] = grouped.transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).max()
    )
    out[f'{out_prefix}_peak_career'] = grouped.transform(
        lambda x: x.shift(1).expanding(min_periods=1).max()
    )
    out[f'{out_prefix}_avg_3yr'] = grouped.transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return out


def add_prior_peak_season_references(df):
    """Attach the prior season with maximum context-adjusted minutes."""
    out = df.sort_values(['player_id', 'season_start']).reset_index(drop=True).copy()
    n_rows = len(out)
    peak_season = np.full(n_rows, np.nan)
    peak_adjusted = np.full(n_rows, np.nan)
    history_count = np.zeros(n_rows, dtype=int)

    for indices in out.groupby('player_id', sort=False).groups.values():
        indices = np.asarray(list(indices), dtype=int)
        adjusted = out.loc[indices, 'final_weighted_minutes'].to_numpy(dtype=float)
        seasons = out.loc[indices, 'season_start'].to_numpy(dtype=float)
        best_position = None

        for position, row_index in enumerate(indices):
            history_count[row_index] = position
            if best_position is not None:
                peak_season[row_index] = seasons[best_position]
                peak_adjusted[row_index] = adjusted[best_position]

            # On ties, retain the more recent peak season.
            if best_position is None or adjusted[position] >= adjusted[best_position]:
                best_position = position

    out['history_seasons_before'] = history_count
    out['peak_season'] = peak_season
    out['peak_adjusted_minutes'] = peak_adjusted
    out['adjusted_vs_peak_season'] = (
        out['final_weighted_minutes'] / out['peak_adjusted_minutes'].replace(0, np.nan)
    )
    return out


def sustained_event_flags(df, ratio):
    """Return adjusted-only candidate, confirmation, first-event, and onset flags."""
    eligible = (
        (df['history_seasons_before'] >= MIN_HISTORY_SEASONS)
        & df['peak_adjusted_minutes'].gt(0)
        & df['peak_adjusted_minutes'].notna()
    )
    low = eligible & (df['adjusted_vs_peak_season'] < ratio)

    previous_low = low.groupby(df['player_id']).shift(1, fill_value=False)
    previous_season = df.groupby('player_id')['season_start'].shift(1)
    consecutive = df['season_start'].eq(previous_season + 1)
    confirmed = low & previous_low & consecutive

    prior_confirmation = confirmed.groupby(df['player_id']).transform(
        lambda x: x.shift(1, fill_value=False).cummax()
    )
    event = confirmed & ~prior_confirmation
    onset = event.groupby(df['player_id']).shift(-1, fill_value=False)
    return low, confirmed, event, onset


print('Loading player-season data...')
ps = pd.read_parquet(OUT_DIR / 'player_season_weighted_minutes.parquet')
ps = ps.sort_values(['player_id', 'season_start']).reset_index(drop=True)

# The weighted-minutes source can retain columns from earlier label revisions.
# Remove those fields so the canonical output cannot silently carry raw-based
# peak or drop-off definitions forward.
deprecated_columns = [
    column for column in ps.columns
    if column.startswith('dropoff_')
    or column.startswith('platform_dropoff_')
    or column.startswith('raw_peak_')
    or column.startswith('cam_peak_')
    or column.startswith('raw_avg')
    or column.startswith('cam_avg')
    or column in {
        'peak_raw_minutes', 'raw_vs_peak_season', 'minutes_vs_peak',
        'injury_associated_dropoff', 'dropoff_comprehensive',
        'dropoff_comprehensive_event', 'dropoff_comprehensive_season',
        'recovery_after_dropoff',
    }
]
ps = ps.drop(columns=deprecated_columns, errors='ignore')

print('Loading injury data...')
injuries = pd.read_csv(DATA_DIR / 'injuries' / 'injuries.csv')
injuries['injury_start_date'] = pd.to_datetime(injuries['injury_start_date'], errors='coerce')
injuries['days_lost'] = pd.to_numeric(injuries['days_lost'], errors='coerce')
injuries['season_start'] = injuries['injury_start_date'].apply(season_start_from_date)

injury_agg = injuries.groupby(['player_id', 'season_start']).agg(
    injury_count=('days_lost', 'count'),
    injury_days_lost=('days_lost', 'sum'),
    severe_injury_count=('days_lost', lambda x: (x >= 28).sum()),
    major_injury_flag=('days_lost', lambda x: int((x >= 90).any())),
).reset_index()
injury_season_min = int(injury_agg['season_start'].min())
injury_season_max = int(injury_agg['season_start'].max())

ps = ps.merge(injury_agg, on=['player_id', 'season_start'], how='left')
ps['injury_data_available'] = ps['season_start'].between(
    injury_season_min, injury_season_max
).astype(int)
injury_cols = ['injury_count', 'injury_days_lost', 'severe_injury_count', 'major_injury_flag']
for col in injury_cols:
    dtype = float if col == 'injury_days_lost' else int
    available = ps['injury_data_available'].eq(1)
    ps.loc[available, col] = ps.loc[available, col].fillna(0)
    if dtype is int:
        ps[col] = ps[col].astype('Int64')

print('Computing rolling and career peak references...')
ps = add_rolling_peak(ps, 'final_weighted_minutes', 'final')
ps = add_rolling_peak(ps, 'club_season_avg_elo', 'club_elo')
ps['injury_disrupted'] = (
    (ps['major_injury_flag'] == 1) | (ps['injury_days_lost'] >= 120)
).fillna(False).astype(int)

ps = add_prior_peak_season_references(ps)
print('Building sustained drop-off labels...')

candidate, confirmed, event, onset = sustained_event_flags(
    ps, PRIMARY_PEAK_RATIO
)

ps['dropoff_candidate'] = candidate.astype(int)
ps['dropoff_confirmed'] = confirmed.astype(int)
ps['dropoff_event'] = event.astype(int)
ps['dropoff_onset'] = onset.astype(int)
ps['complete_career_peak_cohort'] = (
    ps.get('career_history_uncertain', 0) == 0
).astype(int)

event_season = ps.loc[ps['dropoff_event'].eq(1)].set_index('player_id')['season_start']
ps['dropoff_confirmation_season'] = ps['player_id'].map(event_season)
ps['dropoff_onset_season'] = ps['dropoff_confirmation_season'] - 1
confirmed_to_date = ps.groupby('player_id')['dropoff_event'].cummax().astype(bool)
ps['dropoff_status'] = np.select(
    [confirmed_to_date, ps['dropoff_candidate'].eq(1)],
    ['confirmed_dropoff', 'candidate_decline'],
    default='at_risk',
)

previous_injury = ps.groupby('player_id')['injury_disrupted'].shift(1, fill_value=0).astype(bool)
ps['injury_associated_dropoff'] = (
    ps['dropoff_event'].astype(bool)
    & (ps['injury_disrupted'].astype(bool) | previous_injury)
).astype(int)

# Audit whether a confirmed decline is followed by a substantial recovery.
# This is not a second target and never changes the first-event definition.
recovered_players = set()
for player_id, group in ps.groupby('player_id', sort=False):
    event_rows = group[group['dropoff_event'].eq(1)]
    if event_rows.empty:
        continue
    event_row = event_rows.iloc[0]
    event_season_i = int(event_row['season_start'])
    prior_peak = float(event_row['peak_adjusted_minutes'])
    future = group[
        group['season_start'].gt(event_season_i)
        & group['season_start'].le(event_season_i + 3)
    ]
    if prior_peak > 0 and (
        future['final_weighted_minutes'] >= RECOVERY_RATIO * prior_peak
    ).any():
        recovered_players.add(player_id)
ps['recovery_after_dropoff'] = ps['player_id'].isin(
    recovered_players
).astype(int)

threshold_rows = []
for ratio in [0.50, 0.60, 0.65, 0.70]:
    low_i, confirmed_i, event_i, _ = sustained_event_flags(ps, ratio)
    threshold_rows.append({
        'peak_ratio_threshold': ratio,
        'decline_required': 1.0 - ratio,
        'candidate_low_seasons': int(low_i.sum()),
        'confirmed_rows': int(confirmed_i.sum()),
        'first_confirmed_events': int(event_i.sum()),
        'complete_career_cohort_events': int(
            (event_i & ps['complete_career_peak_cohort'].eq(1)).sum()
        ),
    })
threshold_sensitivity = pd.DataFrame(threshold_rows)
event_counts = threshold_sensitivity['first_confirmed_events'].to_numpy()
if not np.all(np.diff(event_counts) >= 0):
    raise RuntimeError(
        "Threshold event counts must be non-decreasing as the retained "
        "peak ratio increases."
    )

previous_candidate = ps.groupby('player_id')['dropoff_candidate'].shift(1)
previous_season = ps.groupby('player_id')['season_start'].shift(1)
invalid_events = ps[
    ps['dropoff_event'].eq(1)
    & ~(
        ps['dropoff_candidate'].eq(1)
        & previous_candidate.eq(1)
        & ps['season_start'].eq(previous_season + 1)
    )
]
if not invalid_events.empty:
    raise RuntimeError(
        f"{len(invalid_events)} events do not have two adjacent low seasons."
    )

print('Building one-record-per-player survival targets...')
survival_records = []
for player_id, group in ps.groupby('player_id'):
    group = group.sort_values('season_start')
    if len(group) < 3:
        continue

    baseline_season = int(group['season_start'].min())
    observed_seasons = set(group['season_start'].astype(int))
    continuous_end = baseline_season
    while continuous_end + 1 in observed_seasons:
        continuous_end += 1
    if continuous_end == baseline_season:
        continue
    event_rows = group[
        group['dropoff_event'].eq(1)
        & group['season_start'].le(continuous_end)
    ]
    if event_rows.empty:
        event_observed = 0
        time_to_event = int(continuous_end - baseline_season)
    else:
        event_observed = 1
        time_to_event = int(event_rows['season_start'].iloc[0] - baseline_season)

    first_row = group.iloc[0]
    survival_records.append({
        'player_id': player_id,
        'baseline_season': baseline_season,
        'event': event_observed,
        'time_to_event': time_to_event,
        'position': first_row['position'],
        'sub_position': first_row['sub_position'],
        'date_of_birth': first_row['date_of_birth'],
        'left_truncated_at_window': first_row.get(
            'left_truncated_at_window', 0
        ),
        'career_history_uncertain': first_row.get(
            'career_history_uncertain', 0
        ),
        'right_censored_at_window_end': first_row.get(
            'right_censored_at_window_end', 0
        ),
        'continuous_observation_end_season': continuous_end,
    })

survival_df = pd.DataFrame(survival_records)

print('\n=== Sustained drop-off diagnostics ===')
print(f'Total player-seasons: {len(ps)}')
print(f"dropoff_candidate: {ps['dropoff_candidate'].sum()} ({ps['dropoff_candidate'].mean():.1%})")
print(f"dropoff_confirmed rows: {ps['dropoff_confirmed'].sum()} ({ps['dropoff_confirmed'].mean():.1%})")
print(f"dropoff_event: {ps['dropoff_event'].sum()} ({ps['dropoff_event'].mean():.1%})")
print(f"injury_associated_dropoff: {ps['injury_associated_dropoff'].sum()}")

print(
    f"recovery_after_dropoff players: {len(recovered_players)} "
    f"({len(recovered_players) / max(int(ps['dropoff_event'].sum()), 1):.1%} "
    "of confirmed events)"
)

print('\nThreshold sensitivity:')
print(threshold_sensitivity.to_string(index=False))
print(f'\nSurvival records: {len(survival_df)}')
print(f"Events observed: {survival_df['event'].sum()} ({survival_df['event'].mean():.1%})")
event_times = survival_df.loc[survival_df['event'].eq(1), 'time_to_event']
print(f'Median time to confirmation: {event_times.median():.1f} seasons')

ps.to_parquet(OUT_DIR / 'player_season_labeled.parquet', index=False)
survival_df.to_parquet(OUT_DIR / 'player_survival_events.parquet', index=False)
threshold_sensitivity.to_csv(REPORT_DIR / 'dropoff_threshold_sensitivity.csv', index=False)
label_checks = {
    'primary_peak_ratio': PRIMARY_PEAK_RATIO,
    'minimum_history_seasons': MIN_HISTORY_SEASONS,
    'player_season_rows': int(len(ps)),
    'eligible_peak_comparisons': int(
        (
            ps['history_seasons_before'].ge(MIN_HISTORY_SEASONS)
            & ps['peak_adjusted_minutes'].gt(0)
        ).sum()
    ),
    'candidate_rows': int(ps['dropoff_candidate'].sum()),
    'confirmed_rows': int(ps['dropoff_confirmed'].sum()),
    'first_confirmed_events': int(ps['dropoff_event'].sum()),
    'recovered_players_within_3yr': int(len(recovered_players)),
    'invalid_event_sequences': int(len(invalid_events)),
    'threshold_event_counts_monotonic': True,
    'adjusted_minutes_only': True,
    'injury_data_season_start': injury_season_min,
    'injury_data_season_end': injury_season_max,
    'complete_career_peak_cohort_players': int(
        ps.loc[
            ps['complete_career_peak_cohort'].eq(1), 'player_id'
        ].nunique()
    ),
    'platform_or_comprehensive_targets_present': any(
        column.startswith(('platform_dropoff_', 'dropoff_comprehensive'))
        for column in ps.columns
    ),
}
(REPORT_DIR / 'adjusted_only_label_checks.json').write_text(
    json.dumps(label_checks, indent=2), encoding='utf-8'
)

print(f"\nSaved: {OUT_DIR / 'player_season_labeled.parquet'}")
print(f"Saved: {OUT_DIR / 'player_survival_events.parquet'}")
print(f"Saved: {REPORT_DIR / 'dropoff_threshold_sensitivity.csv'}")
print(f"Saved: {REPORT_DIR / 'adjusted_only_label_checks.json'}")
