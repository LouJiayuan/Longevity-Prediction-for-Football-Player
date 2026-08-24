"""
Feature engineering for longevity modeling.

Creates two datasets:
1. player_season_features.parquet: time-varying features per player-season for Cox / classification
2. survival_static_features.parquet: static snapshot per player for overall survival prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / 'data' / 'processed'

print("Loading labeled player-season data...")
ps = pd.read_parquet(OUT_DIR / 'player_season_labeled.parquet')

# Sort chronologically per player
ps = ps.sort_values(['player_id', 'season_start']).reset_index(drop=True)

# --- Position grouping ---
def simplify_position(pos, sub_pos):
    if pd.isna(pos):
        return 'Unknown'
    pos = str(pos)
    if pos == 'Goalkeeper':
        return 'Goalkeeper'
    if pos == 'Defender':
        if str(sub_pos) in ['Centre-Back']:
            return 'CentreBack'
        if str(sub_pos) in ['Right-Back', 'Left-Back']:
            return 'FullBack'
        return 'Defender'
    if pos == 'Midfield':
        if str(sub_pos) in ['Defensive Midfield']:
            return 'DefensiveMidfield'
        if str(sub_pos) in ['Central Midfield']:
            return 'CentralMidfield'
        if str(sub_pos) in ['Attacking Midfield']:
            return 'AttackingMidfield'
        return 'Midfield'
    if pos == 'Attack':
        if str(sub_pos) in ['Left Winger', 'Right Winger']:
            return 'Winger'
        if str(sub_pos) in ['Centre-Forward', 'Second Striker']:
            return 'Striker'
        return 'Forward'
    return 'Unknown'

ps['position_group'] = ps.apply(lambda r: simplify_position(r['position'], r['sub_position']), axis=1)

# --- Derive rolling/career features ---
print("Building time-varying features...")

feature_cols = []

# 1. Demographics
ps['age'] = ps['age_at_season_start']
ps['age_sq'] = ps['age'] ** 2
ps['height'] = ps['height_in_cm']
feature_cols += ['age', 'age_sq', 'height']

# 2. Current season context-adjusted exposure
feature_cols += ['final_weighted_minutes', 'games_played']

# 3. Recent form (1-year and 2-year lag)
previous_season = ps.groupby('player_id')['season_start'].shift(1)
two_seasons_back = ps.groupby('player_id')['season_start'].shift(2)
adjacent_lag1 = ps['season_start'].eq(previous_season + 1)
adjacent_lag2 = (
    ps['season_start'].eq(two_seasons_back + 2)
    & adjacent_lag1
)
ps['previous_season_observed'] = adjacent_lag1.fillna(False).astype(int)
feature_cols.append('previous_season_observed')

for col in ['final_weighted_minutes']:
    raw_lag1 = ps.groupby('player_id')[col].shift(1)
    raw_lag2 = ps.groupby('player_id')[col].shift(2)
    ps[f'{col}_lag1'] = raw_lag1.where(adjacent_lag1)
    ps[f'{col}_lag2'] = raw_lag2.where(adjacent_lag2)
    ps[f'{col}_avg2yr'] = pd.concat(
        [ps[f'{col}_lag1'], ps[f'{col}_lag2']], axis=1
    ).mean(axis=1, skipna=True)
    feature_cols += [f'{col}_lag1', f'{col}_lag2', f'{col}_avg2yr']

# Year-over-year change
ps['final_weighted_minutes_yoy'] = ps['final_weighted_minutes'] / ps['final_weighted_minutes_lag1'].clip(lower=1) - 1
feature_cols += ['final_weighted_minutes_yoy']

# 4. Career cumulative features (up to and including current season)
for col in ['final_weighted_minutes', 'games_played']:
    ps[f'career_{col}'] = ps.groupby('player_id')[col].cumsum()
    feature_cols.append(f'career_{col}')

ps['career_seasons'] = ps.groupby('player_id').cumcount() + 1
feature_cols.append('career_seasons')

# Career adjusted minutes per season average
ps['career_adjusted_minutes_per_season'] = (
    ps['career_final_weighted_minutes'] / ps['career_seasons']
)
feature_cols.append('career_adjusted_minutes_per_season')

# 5. Peak-relative features
ps['final_vs_peak'] = ps['final_weighted_minutes'] / ps['final_peak_3yr'].clip(lower=1)
feature_cols += ['final_vs_peak']

# 6. Injury features
for source, output, aggregation in [
    ('injury_count', 'injury_count_2yr', 'sum'),
    ('injury_days_lost', 'injury_days_2yr', 'sum'),
    ('severe_injury_count', 'severe_injury_count_2yr', 'sum'),
    ('major_injury_flag', 'major_injury_flag_2yr', 'max'),
]:
    lag1 = ps.groupby('player_id')[source].shift(1).where(adjacent_lag1)
    lag2 = ps.groupby('player_id')[source].shift(2).where(adjacent_lag2)
    window = pd.concat([lag1, lag2], axis=1)
    if aggregation == 'sum':
        ps[output] = window.sum(axis=1, min_count=1)
    else:
        ps[output] = window.max(axis=1, skipna=True)
feature_cols += ['injury_count', 'injury_days_lost', 'severe_injury_count', 'major_injury_flag',
                 'injury_count_2yr', 'injury_days_2yr', 'severe_injury_count_2yr', 'major_injury_flag_2yr',
                 'injury_data_available']

# 7. Club, match-quality, opponent, and data-quality context
ps['club_elo'] = ps['club_season_avg_elo']
ps['club_elo_vs_peak'] = ps['club_elo'] / ps['club_elo_peak_career'].clip(lower=1)
feature_cols += [
    'club_elo',
    'club_elo_vs_peak',
    'club_rank_percentile',
    'avg_match_quality_elo',
    'avg_match_quality_weight',
    'avg_opponent_elo_gap',
    'avg_opponent_expected_score',
    'avg_opponent_quality_weight',
    'data_quality_score',
    'elo_imputed_minutes_share',
]

competition_share_features = [
    column for column in ps.columns
    if column.endswith('_minutes_share')
    and column != 'elo_imputed_minutes_share'
]
feature_cols += sorted(competition_share_features)

# 8. Position dummies
pos_dummies = pd.get_dummies(ps['position_group'], prefix='pos')
ps = pd.concat([ps, pos_dummies], axis=1)
feature_cols += list(pos_dummies.columns)

# 9. Age × position interaction (for position-specific aging)
for col in pos_dummies.columns:
    ps[f'{col}_x_age'] = ps[col] * ps['age']
    feature_cols.append(f'{col}_x_age')

# 10. Age × cumulative minutes interaction
ps['age_x_career_adjusted_minutes'] = (
    ps['age'] * (ps['career_final_weighted_minutes'] / 10000)
)
feature_cols.append('age_x_career_adjusted_minutes')

# --- Preserve missing values for fold-specific imputation ---
# Global median imputation removed to prevent prospective leakage across temporal folds.

# --- Survival-format dataset (counting process) ---
print("Building survival-format dataset...")

survival_rows = []
for pid, group in ps.groupby('player_id'):
    if len(group) < 2:
        continue
    baseline = group['season_start'].min()
    for i in range(len(group)):
        row = group.iloc[i]
        start = row['season_start'] - baseline
        end = start + 1
        # Stop after the first confirmed sustained event.
        if i > 0 and group.iloc[:i]['dropoff_event'].sum() > 0:
            break
        is_first_dropoff = row['dropoff_event'] == 1
        event = int(is_first_dropoff)
        survival_rows.append({
            'player_id': pid,
            'season_start': row['season_start'],
            'start_time': start,
            'end_time': end,
            'event': event,
            **{c: row[c] for c in feature_cols},
            **{c: row[c] for c in [
                'final_weighted_minutes', 'dropoff_event', 'dropoff_candidate'
            ]},
        })

survival_df = pd.DataFrame(survival_rows)

# --- Binary classification datasets ---
# 1. Fixed-horizon datasets (N=2 and N=5)
print("Building fixed-horizon classification datasets...")

def target_within_calendar_horizon(group, row_index, horizon):
    """Return 1/0/NaN with right-censoring handled explicitly."""
    current_season = int(group.iloc[row_index]['season_start'])
    future_events = group[
        group['dropoff_event'].eq(1)
        & group['season_start'].gt(current_season)
        & group['season_start'].le(current_season + horizon)
    ]
    if not future_events.empty:
        return 1
    observed = set(group['season_start'].astype(int))
    fully_observed = all(
        season in observed
        for season in range(
            current_season + 1, current_season + horizon + 1
        )
    )
    if fully_observed:
        return 0
    return np.nan


def build_classification_dataset(horizon):
    class_rows = []
    for pid, group in ps.groupby('player_id'):
        group = group.sort_values('season_start').reset_index(drop=True)
        for i in range(len(group)):
            # Rows at or after a confirmed event are no longer at risk.
            if group.iloc[:i + 1]['dropoff_event'].sum() > 0:
                continue
            target = target_within_calendar_horizon(group, i, horizon)
            if pd.isna(target):
                continue
            row = group.iloc[i].copy()
            row_dict = {c: row[c] for c in feature_cols}
            row_dict['player_id'] = pid
            row_dict['season_start'] = row['season_start']
            row_dict[f'target_dropoff_next{horizon}yr'] = int(target)
            row_dict['seasons_ahead'] = horizon
            row_dict['horizon'] = horizon
            row_dict['complete_career_peak_cohort'] = row.get(
                'complete_career_peak_cohort', 1
            )
            row_dict['career_history_uncertain'] = row.get(
                'career_history_uncertain', 0
            )
            class_rows.append(row_dict)
    return pd.DataFrame(class_rows)

class_df_2yr = build_classification_dataset(2)
class_df_5yr = build_classification_dataset(5)

print(f"2-year classification dataset: {len(class_df_2yr)} rows, target rate: {class_df_2yr['target_dropoff_next2yr'].mean():.1%}")
print(f"5-year classification dataset: {len(class_df_5yr)} rows, target rate: {class_df_5yr['target_dropoff_next5yr'].mean():.1%}")

# --- Year-by-year cumulative classification dataset ---
print("Building year-by-year cumulative classification dataset...")

yearly_rows = []
for pid, group in ps.groupby('player_id'):
    group = group.sort_values('season_start').reset_index(drop=True)
    for i in range(len(group)):
        if group.iloc[:i + 1]['dropoff_event'].sum() > 0:
            continue
        row_dict = {c: group.iloc[i][c] for c in feature_cols}
        row_dict['player_id'] = pid
        row_dict['season_start'] = group.iloc[i]['season_start']
        current_season = int(group.iloc[i]['season_start'])
        observed = set(group['season_start'].astype(int))
        continuous_ahead = 0
        while current_season + continuous_ahead + 1 in observed:
            continuous_ahead += 1
        row_dict['seasons_ahead'] = min(5, continuous_ahead)
        row_dict['complete_career_peak_cohort'] = group.iloc[i].get(
            'complete_career_peak_cohort', 1
        )
        row_dict['career_history_uncertain'] = group.iloc[i].get(
            'career_history_uncertain', 0
        )

        # cumulative targets: drop-off within k years
        for k in range(1, 6):
            row_dict[f'target_within_{k}yr'] = target_within_calendar_horizon(group, i, k)

        yearly_rows.append(row_dict)

class_df_yearly = pd.DataFrame(yearly_rows)
# Keep rows with at least one estimable horizon.
yearly_target_cols = [f'target_within_{k}yr' for k in range(1, 6)]
class_df_yearly = class_df_yearly[class_df_yearly[yearly_target_cols].notna().any(axis=1)].copy()

print(f"Yearly dataset: {len(class_df_yearly)} rows")
for k in range(1, 6):
    col = f'target_within_{k}yr'
    valid = class_df_yearly[col].notna()
    print(f"  drop-off within {k} year(s): {class_df_yearly.loc[valid, col].mean():.1%} (n={valid.sum()})")

# --- Static snapshot dataset for overall survival ---
print("Building static snapshot dataset...")

# Load survival events with time_to_event from script 02
survival_events = pd.read_parquet(OUT_DIR / 'player_survival_events.parquet')

static_rows = []
for pid, group in ps.groupby('player_id'):
    if len(group) < 3:
        continue
    # Take snapshot at end of 3rd season
    snap = group.iloc[2].copy()
    # Survival outcome
    surv_event = survival_events[(survival_events['player_id'] == pid)]
    if surv_event.empty:
        continue
    surv_event = surv_event.iloc[0]
    row_dict = {c: snap[c] for c in feature_cols}
    row_dict['player_id'] = pid
    row_dict['event'] = surv_event['event']
    row_dict['time_to_event'] = surv_event['time_to_event']
    static_rows.append(row_dict)

static_df = pd.DataFrame(static_rows)

# --- Save ---
ps.to_parquet(OUT_DIR / 'player_season_features.parquet', index=False)
survival_df.to_parquet(OUT_DIR / 'survival_format.parquet', index=False)
class_df_2yr.to_parquet(OUT_DIR / 'classification_format_2yr.parquet', index=False)
class_df_5yr.to_parquet(OUT_DIR / 'classification_format_5yr.parquet', index=False)
class_df_yearly.to_parquet(OUT_DIR / 'classification_format_yearly.parquet', index=False)
static_df.to_parquet(OUT_DIR / 'survival_static_features.parquet', index=False)

print(f"\nSaved: {OUT_DIR / 'player_season_features.parquet'} ({len(ps)} rows)")
print(f"Saved: {OUT_DIR / 'survival_format.parquet'} ({len(survival_df)} rows)")
print(f"Saved: {OUT_DIR / 'classification_format_2yr.parquet'} ({len(class_df_2yr)} rows)")
print(f"Saved: {OUT_DIR / 'classification_format_5yr.parquet'} ({len(class_df_5yr)} rows)")
print(f"Saved: {OUT_DIR / 'classification_format_yearly.parquet'} ({len(class_df_yearly)} rows)")
print(f"Saved: {OUT_DIR / 'survival_static_features.parquet'} ({len(static_df)} rows)")
print(f"\nFeature count: {len(feature_cols)}")
print("Features:", feature_cols[:10], "...", feature_cols[-10:])
