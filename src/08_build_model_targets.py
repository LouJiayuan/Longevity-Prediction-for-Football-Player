"""
Build unified targets for A/B/C modeling.

Outputs:
- survival_landmark.parquet
- regression_format_yearly.parquet
"""

import warnings

import numpy as np
import pandas as pd

from modeling_common import HORIZONS, OUT_DIR

warnings.filterwarnings('ignore')


def build_survival_landmark(ps):
    rows = []
    for pid, group in ps.groupby('player_id'):
        group = group.sort_values('season_start').reset_index(drop=True)
        previous_dropoffs = group['dropoff_event'].shift(1).fillna(0).cumsum()

        for i in range(len(group) - 1):
            row = group.iloc[i].copy()
            if previous_dropoffs.iloc[i] > 0 or row['dropoff_event'] == 1:
                continue

            current_season = int(row['season_start'])
            observed = set(group['season_start'].astype(int))
            continuous_end = current_season
            while continuous_end + 1 in observed:
                continuous_end += 1
            future = group[
                group['season_start'].gt(current_season)
                & group['season_start'].le(continuous_end)
            ].copy()
            if future.empty:
                continue

            events = future[future['dropoff_event'] == 1]
            if events.empty:
                event = 0
                time_to_event = int(future['season_start'].max() - current_season)
            else:
                event = 1
                time_to_event = int(events['season_start'].iloc[0] - current_season)

            out = row.to_dict()
            out['event'] = event
            out['time_to_event'] = max(time_to_event, 1)
            out['observed_future_seasons'] = int(
                continuous_end - current_season
            )
            rows.append(out)

    return pd.DataFrame(rows)


def build_regression_targets(ps):
    rows = []
    for pid, group in ps.groupby('player_id'):
        group = group.sort_values('season_start').reset_index(drop=True)
        by_season = group.set_index('season_start')
        for i in range(len(group) - 1):
            row = group.iloc[i].copy().to_dict()
            current_season = int(group.iloc[i]['season_start'])
            available = 0
            for k in HORIZONS:
                col = f'target_minutes_t_plus_{k}'
                future_season = current_season + k
                if future_season in by_season.index:
                    value = by_season.loc[future_season, 'final_weighted_minutes']
                    row[col] = float(value.iloc[0] if isinstance(value, pd.Series) else value)
                    available += 1
                else:
                    row[col] = np.nan
            row['max_regression_horizon'] = available
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("Loading player-season features...")
    ps = pd.read_parquet(OUT_DIR / 'player_season_features.parquet')
    ps = ps.sort_values(['player_id', 'season_start']).reset_index(drop=True)

    print("Building landmark survival targets...")
    survival = build_survival_landmark(ps)
    if survival.empty:
        raise RuntimeError("survival_landmark target set is empty")
    if survival['time_to_event'].min() < 1:
        raise RuntimeError("survival time_to_event must be >= 1")
    if not set(survival['event'].unique()).issubset({0, 1}):
        raise RuntimeError("survival event must be binary")

    print("Building yearly regression targets...")
    regression = build_regression_targets(ps)
    if regression.empty:
        raise RuntimeError("regression target set is empty")
    target_cols = [f'target_minutes_t_plus_{k}' for k in HORIZONS]
    if regression[target_cols].notna().sum().sum() == 0:
        raise RuntimeError("all regression targets are missing")

    survival_out = OUT_DIR / 'survival_landmark.parquet'
    regression_out = OUT_DIR / 'regression_format_yearly.parquet'
    survival.to_parquet(survival_out, index=False)
    regression.to_parquet(regression_out, index=False)

    print(f"Saved: {survival_out} ({len(survival)} rows)")
    print(f"  events: {survival['event'].sum()} ({survival['event'].mean():.1%})")
    print(f"  median time_to_event: {survival['time_to_event'].median():.1f} seasons")
    print(f"Saved: {regression_out} ({len(regression)} rows)")
    for k in HORIZONS:
        col = f'target_minutes_t_plus_{k}'
        print(f"  {col}: {regression[col].notna().sum()} non-null")


if __name__ == '__main__':
    main()
