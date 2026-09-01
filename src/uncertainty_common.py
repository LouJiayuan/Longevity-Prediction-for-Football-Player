"""Shared temporal-validation and weighted conformal helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / 'config' / 'uncertainty_experiment.json'


def load_uncertainty_config():
    with open(CONFIG_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def add_target_end_season(frame, horizon):
    """Return rows whose full horizon is observable by the data end."""
    config = load_uncertainty_config()
    out = frame.copy()
    out['target_end_season'] = out['season_start'].astype(int) + int(horizon)
    return out[
        out['target_end_season'].le(config['data_end_season'])
    ].copy()


def temporal_windows(frame):
    """Split a target-complete frame for evaluation and deployment."""
    config = load_uncertainty_config()
    year = frame['target_end_season']
    return {
        'selection': frame[
            year.between(
                config['selection_target_year_start'],
                config['selection_target_year_end'],
            )
        ].copy(),
        'evaluation_train': frame[
            year.le(config['selection_target_year_end'])
        ].copy(),
        'evaluation': frame[
            year.between(
                config['evaluation_target_year_start'],
                config['evaluation_target_year_end'],
            )
        ].copy(),
        'final_train': frame[
            year.le(config['final_training_target_year_end'])
        ].copy(),
        'calibration': frame[
            year.between(
                config['calibration_target_year_start'],
                config['calibration_target_year_end'],
            )
        ].copy(),
        'deployment_train': frame[
            year.le(config['deployment_training_target_year_end'])
        ].copy(),
    }


def recency_weights(residual_years, forecast_year, rho):
    residual_years = np.asarray(residual_years, dtype=float)
    age = np.maximum(float(forecast_year) - residual_years, 0.0)
    return np.power(float(rho), age)


def weighted_quantile_with_test_mass(
    values,
    weights,
    quantile,
    test_weight=1.0,
):
    """
    Weighted conformal quantile with the test point represented by +infinity.

    This follows the weighted split-conformal construction used for
    nonexchangeable observations. Returning infinity is intentional when the
    effective calibration mass is too small for the requested quantile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0 or weights.sum() <= 0:
        return np.inf

    augmented_values = np.append(values, np.inf)
    augmented_weights = np.append(weights, float(test_weight))
    order = np.argsort(augmented_values, kind='mergesort')
    augmented_values = augmented_values[order]
    augmented_weights = augmented_weights[order]
    threshold = float(quantile) * augmented_weights.sum()
    index = int(np.searchsorted(
        np.cumsum(augmented_weights), threshold, side='left'
    ))
    return float(augmented_values[min(index, len(augmented_values) - 1)])


def asymmetric_weighted_conformal_offsets(
    residuals,
    residual_years,
    forecast_year,
    coverage,
    rho,
):
    """
    Return lower and upper additive offsets for a two-sided interval.

    residual = observed - predicted. Each tail receives alpha/2, so combining
    the two one-sided bounds gives at least the requested two-sided coverage
    under the corresponding conformal assumptions.
    """
    residuals = np.asarray(residuals, dtype=float)
    weights = recency_weights(residual_years, forecast_year, rho)
    tail_quantile = 1.0 - (1.0 - float(coverage)) / 2.0
    lower_offset = weighted_quantile_with_test_mass(
        -residuals, weights, tail_quantile
    )
    upper_offset = weighted_quantile_with_test_mass(
        residuals, weights, tail_quantile
    )
    return max(lower_offset, 0.0), max(upper_offset, 0.0)


def apply_asymmetric_interval(predictions, lower_offset, upper_offset):
    predictions = np.asarray(predictions, dtype=float)
    lower = np.clip(predictions - lower_offset, 0.0, None)
    upper = np.clip(predictions + upper_offset, 0.0, None)
    return lower, upper


def interval_diagnostics(actual, lower, upper):
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (actual >= lower) & (actual <= upper)
    return {
        'empirical_coverage': float(covered.mean()),
        'mean_interval_width': float(np.mean(upper - lower)),
        'median_interval_width': float(np.median(upper - lower)),
        'n': int(len(actual)),
    }
