#!/usr/bin/env python3
"""Create paper-ready tables and figures for the verified injury comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "outputs" / "verified_injury_feature_comparison"
FIGURE_DIR = BASE_DIR / "figures" / "verified_injury_feature_comparison"

PAIRINGS = (
    ("age_injury", "age_only"),
    ("minutes_injury", "minutes_only"),
    ("age_minutes_elo_injury", "age_minutes_elo"),
    ("full_model_injury", "full_model"),
)


def classification_deltas() -> pd.DataFrame:
    records = []
    for task in ("confirmation", "early_warning"):
        path = (
            RESULT_DIR
            / task
            / "verified_injury_classification_vertical_comparison.csv"
        )
        data = pd.read_csv(path)
        for horizon in (2, 5):
            indexed = data[data["horizon_years"].eq(horizon)].set_index(
                "feature_set"
            )
            for richer, baseline in PAIRINGS:
                records.append({
                    "task": task,
                    "horizon_years": horizon,
                    "injury_feature_set": richer,
                    "baseline_feature_set": baseline,
                    "delta_auc": indexed.loc[richer, "auc"] - indexed.loc[baseline, "auc"],
                    "delta_average_precision": (
                        indexed.loc[richer, "average_precision"]
                        - indexed.loc[baseline, "average_precision"]
                    ),
                    "delta_brier": (
                        indexed.loc[richer, "brier"] - indexed.loc[baseline, "brier"]
                    ),
                })
    return pd.DataFrame(records)


def survival_deltas() -> pd.DataFrame:
    data = pd.read_csv(
        RESULT_DIR / "verified_injury_survival_vertical_comparison.csv"
    ).set_index("feature_set")
    records = []
    for richer, baseline in PAIRINGS:
        records.append({
            "injury_feature_set": richer,
            "baseline_feature_set": baseline,
            "delta_c_index": data.loc[richer, "c_index"] - data.loc[baseline, "c_index"],
            "delta_auc_2yr": data.loc[richer, "auc_2yr"] - data.loc[baseline, "auc_2yr"],
            "delta_auc_5yr": data.loc[richer, "auc_5yr"] - data.loc[baseline, "auc_5yr"],
            "delta_ibs": (
                data.loc[richer, "integrated_brier_score_1to5"]
                - data.loc[baseline, "integrated_brier_score_1to5"]
            ),
        })
    return pd.DataFrame(records)


def classification_auc_figure():
    sets = [
        "age_only",
        "minutes_only",
        "age_minutes_elo",
        "age_minutes_elo_injury",
        "full_model",
        "full_model_injury",
    ]
    labels = [
        "Age",
        "Minutes",
        "Age + Minutes + Elo",
        "+ Injury proxy",
        "Full model",
        "Full + Injury proxy",
    ]
    colors = ["#7A7F87", "#A6ABB2", "#197C83", "#E28A35", "#2E5964", "#C95C3B"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, task in enumerate(("confirmation", "early_warning")):
        data = pd.read_csv(
            RESULT_DIR
            / task
            / "verified_injury_classification_vertical_comparison.csv"
        )
        for column, horizon in enumerate((2, 5)):
            ax = axes[row, column]
            indexed = data[data["horizon_years"].eq(horizon)].set_index("feature_set")
            values = [indexed.loc[name, "auc"] for name in sets]
            positions = np.arange(len(sets))
            ax.barh(positions, values, color=colors, height=0.66)
            ax.set_yticks(positions, labels if column == 0 else [""] * len(labels))
            ax.invert_yaxis()
            ax.set_xlim(0.50, 0.83)
            ax.grid(axis="x", color="#D9DDE1", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_title(
                f"{task.replace('_', ' ').title()} - {horizon} years",
                fontsize=12,
                loc="left",
            )
            for position, value in zip(positions, values):
                ax.text(value + 0.004, position, f"{value:.3f}", va="center", fontsize=9)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Locked classification AUC: verified injury-history extension",
        fontsize=16,
        x=0.06,
        ha="left",
    )
    fig.supxlabel("AUC (0.50 = random ranking)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURE_DIR / "verified_injury_classification_auc.png", dpi=220)
    plt.close(fig)


def incremental_delta_figure(deltas: pd.DataFrame):
    compact = deltas[
        deltas["injury_feature_set"].isin(
            ["age_minutes_elo_injury", "full_model_injury"]
        )
    ].copy()
    compact["task_horizon"] = (
        compact["task"].str.replace("_", " ").str.title()
        + " "
        + compact["horizon_years"].astype(str)
        + "y"
    )
    categories = [
        "Confirmation 2y",
        "Confirmation 5y",
        "Early Warning 2y",
        "Early Warning 5y",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(categories))
    width = 0.34
    for offset, (feature_set, label, color) in enumerate([
        ("age_minutes_elo_injury", "Age + Minutes + Elo: injury increment", "#E28A35"),
        ("full_model_injury", "Full model: injury increment", "#C95C3B"),
    ]):
        indexed = compact[compact["injury_feature_set"].eq(feature_set)].set_index(
            "task_horizon"
        )
        values = [indexed.loc[category, "delta_auc"] for category in categories]
        positions = x + (offset - 0.5) * width
        bars = ax.bar(positions, values, width=width, color=color, label=label)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset_y = 0.00025 if value >= 0 else -0.00025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset_y,
                f"{value:+.3f}",
                ha="center",
                va=va,
                fontsize=9,
            )
    ax.axhline(0, color="#30343A", linewidth=1)
    ax.set_xticks(x, categories)
    ax.set_ylabel("AUC difference after adding injury proxy")
    ax.set_title("Incremental value is concentrated at the 5-year horizon", loc="left")
    ax.grid(axis="y", color="#D9DDE1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "verified_injury_incremental_auc.png", dpi=220)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    deltas = classification_deltas()
    survival = survival_deltas()
    deltas.to_csv(RESULT_DIR / "verified_injury_classification_deltas.csv", index=False)
    survival.to_csv(RESULT_DIR / "verified_injury_survival_deltas.csv", index=False)
    classification_auc_figure()
    incremental_delta_figure(deltas)
    print(deltas.to_string(index=False))
    print(survival.to_string(index=False))


if __name__ == "__main__":
    main()
