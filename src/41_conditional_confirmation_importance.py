"""Conditional confirmation, and matched feature importance for all three tasks.

Motivation. The unconditional confirmation classifier scores AUC 0.967 at a
one-year horizon, but that number is not forecasting: by construction every
one-year positive is a landmark whose current season is already low, because a
player is removed from the risk set once confirmed. The interesting question is
the conditional one the deployment layer already asks - given that a player is
*in* a first low season now, will the next adjacent season also stay below the
threshold? - and script 37 answers it (`next_season_confirmation`, AUC 0.654).

What script 37 does not produce is feature importance, so there is no way to say
*why* the three tasks differ. This script fills that gap and makes the three
comparable by holding two things fixed:

  * the feature set: `age_minutes_elo` (22 features) for every arm, which is
    already the deployment feature set and already the selected set for
    confirmation; and
  * the algorithm: XGBoost, the reference algorithm used for importance in
    script 40.

Three arms, all under script 37's deployment protocol so that they share one
temporal rule:

  A. Early warning, h = 1..5. The player is currently stable; predict whether a
     first low season belonging to a subsequently confirmed decline begins
     within h years.
  B. Conditional confirmation, h = 1. The player is currently in a first low
     season; predict whether the next adjacent season also stays low.
  C. Unconditional confirmation, h = 1 and h = 2, rebuilt here under the same
     deployment protocol purely so that arm C's importance is measured the same
     way as A and B. Its AUC therefore does not reproduce the locked tables and
     is written out clearly labelled; the locked numbers remain those of
     scripts 29 and 40.

Each arm is also refitted with a one-feature age-only specification under the
identical protocol and algorithm, so that "the model beats age by X" is a
like-for-like statement rather than a comparison against a raw age ranking.

Verification gate. Arms A and B must reproduce
`outputs/practical_application/sequential_model_metrics.csv` exactly, because
they reuse script 37's own functions on the same inputs. The gate is written to
`verification_vs_sequential_metrics.csv` and the script refuses to write
importance if it fails.

Run from the workspace root:

    python src/41_conditional_confirmation_importance.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "sequential_application", SCRIPTS / "37_sequential_player_application.py")
sq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sq)

ROOT = SCRIPTS.parent
DATA_ROOT = ROOT / "data"
DATA = DATA_ROOT / "processed"
OUT = ROOT / "outputs" / "conditional_confirmation"
OUT.mkdir(parents=True, exist_ok=True)

PERMUTATION_REPEATS = 5
SEED = 42
GATE_TOLERANCE = 1e-9


def evaluation_split(frame):
    """Script 37's locked split: fit on <= 2021 target ends, score 2022-2023."""
    train = frame[frame["target_end_season"].le(
        max(sq.SELECTION_TARGET_YEARS))].copy()
    test = frame[frame["target_end_season"].isin(
        sq.EVALUATION_TARGET_YEARS)].copy()
    return train, test


def arm_metrics_and_importance(frame, features, task, horizon):
    """Refit on the locked split, then permute each feature on the test rows."""
    train, test = evaluation_split(frame)
    if train.empty or test.empty or train["_target"].nunique() < 2:
        raise SystemExit(f"{task} h={horizon}: empty or single-class split.")

    # Calibration is fitted exactly as in script 37, from selection-period
    # out-of-fold predictions, so the reported AUC matches its metrics file.
    oof = sq.temporal_oof(frame, features)
    calibrator = sq.fit_platt(oof["target"], oof["raw_probability"])
    model, raw, used, _ = sq.fit_predict(train, test, features)
    calibrated = sq.apply_platt(calibrator, raw)

    y = test["_target"].astype(int).to_numpy()
    metrics = {"task": task, "horizon_years": int(horizon),
               "feature_set": "age_minutes_elo",
               **sq.binary_metrics(y, calibrated)}

    # Importance is measured on the ranking, so it uses the uncalibrated score;
    # Platt is monotone and cannot change an AUC.
    _, X_test, _, _, _ = sq.clean_feature_matrix(train, test, test, features)
    base_auc = float(roc_auc_score(y, raw))
    rng = np.random.default_rng(SEED)
    rows = []
    for column in used:
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X_test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            p = model.predict_proba(shuffled)[:, 1]
            drops.append(base_auc - float(roc_auc_score(y, p)))
        rows.append({"task": task, "horizon_years": int(horizon),
                     "feature": column,
                     "auc_decrease_mean": float(np.mean(drops)),
                     "auc_decrease_std": float(np.std(drops)),
                     "baseline_auc": base_auc,
                     "n_test": int(len(y)),
                     "event_rate": float(y.mean())})
    return metrics, rows


def arm_metrics_only(frame, features, task, horizon, label):
    """Same protocol, a different feature list; used for the age-only baseline."""
    train, test = evaluation_split(frame)
    oof = sq.temporal_oof(frame, features)
    calibrator = sq.fit_platt(oof["target"], oof["raw_probability"])
    _, raw, _, _ = sq.fit_predict(train, test, features)
    calibrated = sq.apply_platt(calibrator, raw)
    y = test["_target"].astype(int).to_numpy()
    return {"task": task, "horizon_years": int(horizon), "feature_set": label,
            **sq.binary_metrics(y, calibrated)}


def build_early_warning(ew_yearly, horizon, data_end):
    target = f"target_within_{horizon}yr"
    frame = ew_yearly.dropna(subset=[target]).copy()
    frame["_target"] = frame[target].astype(int)
    frame["target_end_season"] = frame["season_start"].astype(int) + horizon
    frame["followup_end_season"] = frame["target_end_season"] + 1
    return frame[frame["followup_end_season"].le(data_end)].copy()


def build_unconditional_confirmation(conf_yearly, horizon, data_end):
    """Arm C: the ordinary confirmation target, on the deployment protocol."""
    target = f"target_within_{horizon}yr"
    frame = conf_yearly.dropna(subset=[target]).copy()
    frame["_target"] = frame[target].astype(int)
    frame["target_end_season"] = frame["season_start"].astype(int) + horizon
    return frame[frame["target_end_season"].le(data_end)].copy()


def main() -> None:
    config = json.loads((ROOT / "config/uncertainty_experiment.json").read_text())
    data_end = int(config["data_end_season"])

    features = pd.read_parquet(DATA / "player_season_features.parquet")
    ew_yearly = pd.read_parquet(DATA / "classification_early_warning_yearly.parquet")
    conf_yearly = pd.read_parquet(DATA / "classification_confirmation_yearly.parquet")
    model_features = sq.age_minutes_elo_features(set(features.columns))
    print(f"feature set: age_minutes_elo ({len(model_features)} features)")

    metric_rows, importance_rows = [], []

    print("\nArm A - early warning, h = 1..5 (currently stable players)")
    for horizon in sq.HORIZONS:
        frame = build_early_warning(ew_yearly, horizon, data_end)
        m, imp = arm_metrics_and_importance(
            frame, model_features, "early_warning", horizon)
        metric_rows.append(m)
        importance_rows.extend(imp)
        base = arm_metrics_only(frame, ["age"], "early_warning", horizon,
                                "age_only")
        metric_rows.append(base)
        print(f"   h={horizon}  AUC {m['auc']:.4f}  (age only {base['auc']:.4f}, "
              f"gain {m['auc']-base['auc']:+.4f})  n={m['n']:,}  "
              f"event rate {m['event_rate']*100:.1f}%")

    print("\nArm B - conditional confirmation, h = 1 (players already in a "
          "first low season)")
    conditional = sq.build_next_season_confirmation(features)
    m, imp = arm_metrics_and_importance(
        conditional, model_features, "next_season_confirmation", 1)
    metric_rows.append(m)
    importance_rows.extend(imp)
    base = arm_metrics_only(conditional, ["age"], "next_season_confirmation", 1,
                            "age_only")
    metric_rows.append(base)
    print(f"   h=1  AUC {m['auc']:.4f}  (age only {base['auc']:.4f}, "
          f"gain {m['auc']-base['auc']:+.4f})  n={m['n']:,}  "
          f"event rate {m['event_rate']*100:.1f}%")

    print("\nArm C - unconditional confirmation, h = 1 and 2 "
          "(deployment protocol; NOT the locked numbers)")
    for horizon in (1, 2):
        frame = build_unconditional_confirmation(conf_yearly, horizon, data_end)
        m, imp = arm_metrics_and_importance(
            frame, model_features, "confirmation_unconditional", horizon)
        metric_rows.append(m)
        importance_rows.extend(imp)
        base = arm_metrics_only(frame, ["age"], "confirmation_unconditional",
                                horizon, "age_only")
        metric_rows.append(base)
        print(f"   h={horizon}  AUC {m['auc']:.4f}  (age only {base['auc']:.4f}, "
              f"gain {m['auc']-base['auc']:+.4f})  n={m['n']:,}  "
              f"event rate {m['event_rate']*100:.1f}%")

    metrics = pd.DataFrame(metric_rows)

    # ---------------------------------------------------------------- the gate
    published = pd.read_csv(
        ROOT / "outputs/practical_application/sequential_model_metrics.csv")
    checks = []
    for _, row in published.iterrows():
        mine = metrics[(metrics["task"] == row["task"])
                       & (metrics["horizon_years"] == row["horizon_years"])
                       & (metrics["feature_set"] == "age_minutes_elo")]
        if mine.empty:
            continue
        checks.append({"task": row["task"],
                       "horizon_years": int(row["horizon_years"]),
                       "published_auc": float(row["auc"]),
                       "recomputed_auc": float(mine["auc"].iloc[0]),
                       "abs_auc_difference": abs(
                           float(row["auc"]) - float(mine["auc"].iloc[0]))})
    gate = pd.DataFrame(checks)
    gate.to_csv(OUT / "verification_vs_sequential_metrics.csv", index=False)
    worst = float(gate["abs_auc_difference"].max())
    print(f"\nverification gate: {len(gate)} comparisons, "
          f"largest |dAUC| = {worst:.2e}")
    if worst > GATE_TOLERANCE:
        raise SystemExit(
            "Arms A and B do not reproduce sequential_model_metrics.csv. "
            "The protocol is wrong; importance was not written.")
    print("   gate passed")

    metrics.to_csv(OUT / "task_comparison_metrics.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        OUT / "task_comparison_permutation_importance.csv", index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}/task_comparison_metrics.csv")
    print(f"wrote {OUT.relative_to(ROOT)}/"
          "task_comparison_permutation_importance.csv")


if __name__ == "__main__":
    main()
