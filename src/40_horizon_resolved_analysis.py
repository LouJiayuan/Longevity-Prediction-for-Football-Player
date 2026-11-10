"""Horizon-resolved analysis: how performance and feature importance change
with the prediction horizon.

The locked experiment (`29_final_experiment.py`) evaluates classification at two
points only, h = 2 and h = 5, and reports time-dependent survival AUC at the same
two times. This script fills in the intermediate years so the horizon can be read
as a curve rather than two points.

Two analyses, both strictly descriptive:

A. **Survival across t = 1..5.** Each survival model is fitted *once* on the
   locked training cohort and then evaluated at five follow-up times. Nothing is
   refitted per time point, so the curve shows how a single fixed risk ranking
   discriminates events occurring by progressively later times, alongside a
   genuinely time-varying Brier score. Time-resolved permutation importance is
   computed for the pre-selected random survival forest.

B. **Classification across h = 1..5.** For each task the manifest's selected
   feature set is held fixed and every candidate algorithm is refitted at each
   horizon under the same temporal rule used by script 29 (target end = season +
   h, complete follow-up by 2025, selection folds ending 2018-2021, locked test
   2022-2023, Platt calibration from selection-period out-of-fold predictions).
   Permutation importance is computed at each horizon for one fixed reference
   algorithm so that the horizon is the only thing varying.

Nothing here reselects a feature set or an algorithm. The h = 2 and h = 5 rows
reproduce the locked tables exactly and are written out as a verification check.

Run from the workspace root:

    python src/40_horizon_resolved_analysis.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "final_experiment", SCRIPTS / "29_final_experiment.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

from sksurv.metrics import brier_score, cumulative_dynamic_auc  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from uncertainty_common import (  # noqa: E402
    add_target_end_season,
    load_uncertainty_config,
    temporal_windows,
)

CONFIG = load_uncertainty_config()

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs" / "horizon_resolved"
OUT.mkdir(parents=True, exist_ok=True)

TIMES = (1.0, 2.0, 3.0, 4.0, 5.0)
SWEEP_HORIZONS = (1, 2, 3, 4, 5)
REFERENCE_ALGORITHM = "xgboost"
PERMUTATION_REPEATS = 5
SEED = 42


# --------------------------------------------------------------------------
# A. Survival across t = 1..5
# --------------------------------------------------------------------------

def survival_time_curves() -> None:
    frame = pd.read_parquet(
        ROOT / "data/processed/survival_landmark.parquet")
    manifest = json.loads(
        (ROOT / "outputs/confirmation/final_experiment_manifest.json")
        .read_text())
    feature_set = manifest["selected_survival_feature_set"]
    features = manifest["feature_sets"][feature_set]
    preselected = manifest["selected_survival_model"]

    train = fx.administrative_training(
        frame, fx.SURVIVAL_EVALUATION_TRAIN_CUTOFF)
    test = fx.administrative_test_cohort(
        frame, 2019, 2020, fx.SURVIVAL_EVALUATION_OUTCOME_CUTOFF)
    y_train, y_test = fx.to_surv(train), fx.to_surv(test)
    times = np.asarray(TIMES, dtype=float)

    print(f"A. survival: {len(train):,} train / {len(test):,} test rows, "
          f"{int(test['event'].sum()):,} events, feature set {feature_set}")

    def curve(risk, surv):
        """AUC(t) from the scalar risk marker and Brier(t) from S(t|x)."""
        row = {}
        for i, t in enumerate(times):
            try:
                _, auc = cumulative_dynamic_auc(
                    y_train, y_test, np.asarray(risk, float),
                    np.asarray([t]))
                row[f"auc_{int(t)}yr"] = float(auc)
            except Exception:
                row[f"auc_{int(t)}yr"] = np.nan
        try:
            _, bs = brier_score(y_train, y_test, surv, times)
            for t, b in zip(times, bs):
                row[f"brier_{int(t)}yr"] = float(b)
        except Exception:
            for t in times:
                row[f"brier_{int(t)}yr"] = np.nan
        return row

    records, fitted_store = [], {}
    for model_name in fx.SURVIVAL_MODELS:
        print(f"   fitting {model_name} ...", flush=True)
        fitted, _, _, _, X_test_model, used, _ = fx.fit_survival(
            train, test, features, model_name, selection=False)
        risk, surv = fx.survival_predictions(
            fitted, model_name, X_test_model, list(times))
        records.append({
            "feature_set": feature_set,
            "model": model_name,
            "preselected": model_name == preselected,
            "n_features": len(used),
            "c_index": float(fx.concordance_index_censored(
                test["event"].astype(bool).to_numpy(),
                test["time_to_event"].astype(float).to_numpy(),
                np.asarray(risk, float))[0]),
            **curve(risk, surv),
            "n": len(test),
            "events": int(test["event"].sum()),
        })
        fitted_store[model_name] = (fitted, X_test_model, used)

    pd.DataFrame(records).to_csv(
        OUT / "survival_time_resolved_metrics.csv", index=False)
    print(f"   wrote survival_time_resolved_metrics.csv")

    # Time-resolved permutation importance for the pre-selected model.
    fitted, X_test_model, used = fitted_store[preselected]

    def risk_only(X):
        """Risk score alone. AUC(t) needs only the ranking, and skipping the
        survival-function evaluation makes permutation importance tractable."""
        if preselected == "weibull_aft":
            model, _ = fitted
            return -model.predict_expectation(X).to_numpy()
        if preselected == "cox_ph":
            return np.asarray(fitted.predict(X), dtype=float)
        return np.asarray(fitted.predict(X), dtype=float)

    base = {}
    risk = risk_only(X_test_model)
    for t in times:
        _, auc = cumulative_dynamic_auc(
            y_train, y_test, np.asarray(risk, float), np.asarray([t]))
        base[t] = float(auc)

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"   permutation importance for {preselected} "
          f"({len(used)} features x {PERMUTATION_REPEATS} repeats) ...",
          flush=True)
    for column in used:
        drops = {t: [] for t in times}
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X_test_model.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            r = risk_only(shuffled)
            for t in times:
                try:
                    _, auc = cumulative_dynamic_auc(
                        y_train, y_test, np.asarray(r, float),
                        np.asarray([t]))
                    drops[t].append(base[t] - float(auc))
                except Exception:
                    drops[t].append(np.nan)
        for t in times:
            rows.append({
                "model": preselected,
                "time_years": int(t),
                "feature": column,
                "auc_decrease_mean": float(np.nanmean(drops[t])),
                "auc_decrease_std": float(np.nanstd(drops[t])),
                "baseline_auc": base[t],
            })
    pd.DataFrame(rows).to_csv(
        OUT / "survival_time_resolved_permutation_importance.csv", index=False)
    print("   wrote survival_time_resolved_permutation_importance.csv")


# --------------------------------------------------------------------------
# B. Classification across h = 1..5
# --------------------------------------------------------------------------

def _selection_oof(frame, features, model_name, horizon):
    """Out-of-fold selection-period predictions, as script 29 builds them."""
    target = f"target_within_{horizon}yr"
    data = frame.dropna(subset=[target]).copy()
    data["_target"] = data[target].astype(int)
    data = add_target_end_season(data, horizon)
    pieces = []
    for target_year in range(CONFIG["selection_target_year_start"],
                             CONFIG["selection_target_year_end"] + 1):
        train = data[data["target_end_season"].lt(target_year)]
        test = data[data["target_end_season"].eq(target_year)]
        if train.empty or test.empty or train["_target"].nunique() < 2:
            continue
        _, raw, _, _ = fx.fit_classification(train, test, features, model_name)
        pieces.append(pd.DataFrame({
            "target": test["_target"].to_numpy(),
            "raw_probability": raw,
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else None


def classification_horizon_sweep() -> None:
    metric_rows, importance_rows = [], []
    for task in ("confirmation", "early_warning"):
        manifest = json.loads(
            (ROOT / f"outputs/{task}/final_experiment_manifest.json")
            .read_text())
        entry = manifest["classification_tasks"][task]
        feature_set = entry["selected_feature_set"]
        features = manifest["feature_sets"][feature_set]
        preselected = {int(k): v for k, v in entry["selected_models"].items()}
        frame = pd.read_parquet(
            ROOT / "data/processed" / fx.CLASSIFICATION_TASK_FILES[task])

        print(f"B. {task}: feature set {feature_set} "
              f"({len(features)} declared)")

        for horizon in SWEEP_HORIZONS:
            target = f"target_within_{horizon}yr"
            if target not in frame.columns:
                print(f"   h={horizon}: no {target} column, skipped")
                continue
            data = frame.dropna(subset=[target]).copy()
            data["_target"] = data[target].astype(int)
            data = add_target_end_season(data, horizon)
            windows = temporal_windows(data)
            train, test = windows["evaluation_train"], windows["evaluation"]
            if train.empty or test.empty:
                print(f"   h={horizon}: empty window, skipped")
                continue

            for model_name in fx.CLASSIFICATION_MODELS:
                oof = _selection_oof(frame, features, model_name, horizon)
                _, raw, used, _ = fx.fit_classification(
                    train, test, features, model_name)
                if oof is not None:
                    calibrator = fx.fit_platt(
                        oof["target"], oof["raw_probability"])
                    prob = fx.apply_platt(calibrator, raw)
                    calibrated = True
                else:
                    prob, calibrated = raw, False
                y = test["_target"].astype(int)
                metric_rows.append({
                    "task": task,
                    "horizon_years": horizon,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_features": len(used),
                    "preselected_at_this_horizon":
                        preselected.get(horizon) == model_name,
                    "calibrated": calibrated,
                    "auc": float(roc_auc_score(y, prob)),
                    "average_precision": float(average_precision_score(y, prob)),
                    "brier": float(brier_score_loss(y, prob)),
                    "n": len(test),
                    "event_rate": float(y.mean()),
                })
            print(f"   h={horizon}: n={len(test):,} "
                  f"event rate={test['_target'].mean():.3f}", flush=True)

            # Permutation importance with the algorithm held fixed, so the
            # horizon is the only thing that changes between panels.
            model, raw, used, _ = fx.fit_classification(
                train, test, features, REFERENCE_ALGORITHM)
            X_train, X_test, _, _, _ = fx.clean_feature_matrix(
                train, test, test, features)
            y = test["_target"].astype(int)
            base_auc = float(roc_auc_score(y, raw))
            rng = np.random.default_rng(SEED)
            for column in used:
                drops = []
                for _ in range(PERMUTATION_REPEATS):
                    shuffled = X_test.copy()
                    shuffled[column] = rng.permutation(
                        shuffled[column].to_numpy())
                    p = model.predict_proba(shuffled)[:, 1]
                    drops.append(base_auc - float(roc_auc_score(y, p)))
                importance_rows.append({
                    "task": task,
                    "horizon_years": horizon,
                    "model": REFERENCE_ALGORITHM,
                    "feature": column,
                    "auc_decrease_mean": float(np.mean(drops)),
                    "auc_decrease_std": float(np.std(drops)),
                    "baseline_auc": base_auc,
                })

    pd.DataFrame(metric_rows).to_csv(
        OUT / "classification_horizon_metrics.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        OUT / "classification_horizon_permutation_importance.csv", index=False)
    print("   wrote classification_horizon_metrics.csv and "
          "classification_horizon_permutation_importance.csv")


def verification() -> None:
    """The h = 2 and h = 5 sweep rows must reproduce the locked tables."""
    sweep = pd.read_csv(OUT / "classification_horizon_metrics.csv")
    checks = []
    for task in ("confirmation", "early_warning"):
        locked = pd.read_csv(
            ROOT / f"outputs/{task}/final_classification_model_comparison.csv")
        for _, r in locked.iterrows():
            m = sweep[(sweep.task == task)
                      & (sweep.horizon_years == r.horizon_years)
                      & (sweep.model == r.model)]
            if m.empty:
                checks.append((task, r.horizon_years, r.model, np.nan))
                continue
            checks.append((task, int(r.horizon_years), r.model,
                           abs(float(m.auc.iloc[0]) - float(r.auc))))
    frame = pd.DataFrame(
        checks, columns=["task", "horizon_years", "model", "abs_auc_difference"])
    frame.to_csv(OUT / "verification_vs_locked_tables.csv", index=False)
    worst = frame["abs_auc_difference"].max()
    print(f"\nVerification: largest |AUC difference| vs the locked tables at "
          f"h=2 and h=5 is {worst:.2e}")
    if worst > 1e-9:
        print("  WARNING: the sweep does not exactly reproduce the locked "
              "tables. Check the temporal protocol before citing it.")


def main() -> None:
    survival_time_curves()
    print()
    classification_horizon_sweep()
    verification()
    print(f"\nOutputs in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
