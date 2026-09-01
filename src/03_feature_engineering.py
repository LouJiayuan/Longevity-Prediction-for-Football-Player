"""Build player-season features and the two classification landmark tables.

The current survival landmark is built by ``08_build_model_targets.py``.
Legacy counting-process, static-survival and compatibility classification
tables are no longer generated.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from classification_targets import (
    CLASSIFICATION_TASKS,
    assert_cumulative_targets,
    build_yearly_classification_dataset,
)
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

# 6. Club, match-quality, opponent, and data-quality context
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

# 7. Position dummies
pos_dummies = pd.get_dummies(ps['position_group'], prefix='pos')
ps = pd.concat([ps, pos_dummies], axis=1)
feature_cols += list(pos_dummies.columns)

# 8. Age × position interaction (for position-specific aging)
for col in pos_dummies.columns:
    ps[f'{col}_x_age'] = ps[col] * ps['age']
    feature_cols.append(f'{col}_x_age')

# 9. Age × cumulative minutes interaction
ps['age_x_career_adjusted_minutes'] = (
    ps['age'] * (ps['career_final_weighted_minutes'] / 10000)
)
feature_cols.append('age_x_career_adjusted_minutes')

# --- Preserve missing values for fold-specific imputation ---
# Global median imputation removed to prevent prospective leakage across temporal folds.

# --- Binary classification datasets ---
print("Building confirmation and early-warning classification datasets...")
classification_by_task = {}
for prediction_task in CLASSIFICATION_TASKS:
    task_frame = build_yearly_classification_dataset(
        ps, feature_cols, prediction_task
    )
    assert_cumulative_targets(task_frame)
    classification_by_task[prediction_task] = task_frame
    print(f"{prediction_task}: {len(task_frame)} rows")
    for k in range(1, 6):
        col = f'target_within_{k}yr'
        valid = task_frame[col].notna()
        print(
            f"  event within {k} year(s): "
            f"{task_frame.loc[valid, col].mean():.1%} (n={valid.sum()})"
        )

class_df_yearly = classification_by_task['confirmation']
class_df_early_warning = classification_by_task['early_warning']

# --- Save ---
ps.to_parquet(OUT_DIR / 'player_season_features.parquet', index=False)
class_df_yearly.to_parquet(
    OUT_DIR / 'classification_confirmation_yearly.parquet', index=False
)
class_df_early_warning.to_parquet(
    OUT_DIR / 'classification_early_warning_yearly.parquet', index=False
)

print(f"\nSaved: {OUT_DIR / 'player_season_features.parquet'} ({len(ps)} rows)")
print(
    f"Saved: {OUT_DIR / 'classification_confirmation_yearly.parquet'} "
    f"({len(class_df_yearly)} rows)"
)
print(
    f"Saved: {OUT_DIR / 'classification_early_warning_yearly.parquet'} "
    f"({len(class_df_early_warning)} rows)"
)
print(f"\nFeature count: {len(feature_cols)}")
print("Features:", feature_cols[:10], "...", feature_cols[-10:])
