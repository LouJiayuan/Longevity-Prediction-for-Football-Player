"""
Shared helpers for the A/B/C longevity modeling pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'outputs'
FIG_DIR = BASE_DIR / 'figures'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 4, 5]
TRAIN_CUTOFF = 2019

IDENTIFIER_COLS = {
    'player_id', 'season_start', 'player_club_id', 'player_domestic_comp',
    'first_date', 'last_date', 'date_of_birth'
}
TEXT_COLS = {
    'name', 'position', 'sub_position', 'foot', 'position_group',
    'dropoff_status', 'prediction_task',
}
TARGET_PREFIXES = ('target_',)
DROP_OFF_COLS = set()
LEAKAGE_FEATURES = {
    'final_vs_peak',
    'club_elo_vs_peak',
    'peak_season',
    'peak_adjusted_minutes',
    'adjusted_vs_peak_season',
    'recovery_after_dropoff',
}
AUDIT_ONLY_FEATURES = {
    'total_minutes',
    'cam_minutes',
    'left_truncated_at_window',
    'career_history_uncertain',
    'complete_career_peak_cohort',
    'right_censored_at_window_end',
    'first_scope_season_observed',
    'last_scope_season_observed',
    'age_at_first_scope_season',
    'first_club_season_api',
    'last_club_season_api',
    'has_pre_window_club_appearance',
    'has_post_window_club_appearance',
}


def base_feature_columns(df, target_cols=None, extra_exclude=None):
    """Return numeric model features shared by all forward-looking models."""
    target_cols = set(target_cols or [])
    extra_exclude = set(extra_exclude or [])
    excluded = (
        IDENTIFIER_COLS | TEXT_COLS | DROP_OFF_COLS | LEAKAGE_FEATURES |
        AUDIT_ONLY_FEATURES |
        target_cols | extra_exclude
    )

    features = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.startswith(('dropoff_', 'legacy_dropoff_')):
            continue
        # The legacy injury-event file was a synthetic test fixture. Exclude
        # stale injury fields defensively even if an older processed table is
        # supplied to the corrected final experiment.
        if col.startswith('injury_'):
            continue
        if col.startswith(TARGET_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def clean_feature_matrix(train_df, test_df, score_df, feature_cols):
    """
    Median-impute numeric feature matrices using train-set medians and remove
    constant train-set columns.
    """
    medians = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).median()
    medians = medians.fillna(0.0)

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
    variable_cols = [c for c in feature_cols if X_train[c].nunique() > 1]

    X_train = X_train[variable_cols]
    X_test = test_df[variable_cols].replace([np.inf, -np.inf], np.nan).fillna(medians[variable_cols])
    X_score = score_df[variable_cols].replace([np.inf, -np.inf], np.nan).fillna(medians[variable_cols])

    return X_train, X_test, X_score, variable_cols, medians[variable_cols]


def latest_player_rows(ps, at_risk_only=False, current_season_only=False):
    """
    Return each player's latest row.

    When current_season_only is true, retain only players whose latest observed
    season is the latest season in the modeling data. This prevents retired or
    otherwise stale historical rows from being presented as current forecasts.
    """
    latest_season = int(ps['season_start'].max())
    latest = (
        ps.sort_values(['player_id', 'season_start'])
        .groupby('player_id', as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if at_risk_only and 'dropoff_status' in latest.columns:
        latest = latest[latest['dropoff_status'].ne('confirmed_dropoff')].reset_index(drop=True)
    if current_season_only:
        latest = latest[
            latest['season_start'].eq(latest_season)
        ].reset_index(drop=True)
    return latest


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(y_true - y_pred)
    wape_denom = np.abs(y_true).sum()
    smape_denom = np.abs(y_true) + np.abs(y_pred)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': rmse(y_true, y_pred),
        'r2': float(r2_score(y_true, y_pred)),
        'wape': float(
            absolute_error.sum() / wape_denom
        ) if wape_denom > 0 else np.nan,
        'smape': float(
            np.mean(
                np.divide(
                    2.0 * absolute_error,
                    smape_denom,
                    out=np.zeros_like(absolute_error),
                    where=smape_denom > 0,
                )
            )
        ),
    }


def enforce_probability_bounds(values):
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
