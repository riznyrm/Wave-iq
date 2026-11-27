"""Rebuild presentation figures exclusively from the frozen run outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"dummy_median": "Median baseline", "ridge": "Ridge",
          "decision_tree": "Decision tree", "random_forest": "Random forest"}
NAVY, BLUE, ORANGE, GRAY = "#183650", "#197FA5", "#D77838", "#6C7A89"


def create_plots(project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    reports = root / "reports"
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(reports / "cv_comparison.csv").set_index("model")
    holdout = pd.read_csv(reports / "holdout_comparison.csv").set_index("model")
    preds = pd.read_csv(reports / "holdout_predictions.csv")
    selection = json.loads((root / "artifacts/selection.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "data/processed/data_audit.json").read_text(encoding="utf-8"))
    chosen = selection["selected_predictor"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": NAVY, "text.color": NAVY, "axes.titleweight": "bold",
        "axes.titlesize": 13, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    saved = []

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    names = list(LABELS)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="constrained")
    ax.barh(positions - .18, cv.loc[names, "cv_MAE"], height=.33, color=BLUE,
            xerr=cv.loc[names, "cv_MAE_std"], error_kw={"ecolor": NAVY, "capsize": 3},
            label="Development CV mean and fold standard deviation")
    ax.barh(positions + .18, holdout.loc[names, "MAE"], height=.33,
            color=ORANGE, label="Frozen test set")
    ax.set_yticks(positions, [LABELS[n] + ("  [selected]" if n == chosen else "") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("MAE in recorded rejection units (lower is better)")
    ax.set_title("Model errors on unseen order families", loc="left", pad=15)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    save(fig, "model_comparison.png")

    actual = preds["actual_Rejection"].to_numpy()
    predicted = preds[f"predicted_{chosen}"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), layout="constrained")
    for axis, transform, title in [(axes[0], lambda a:a, "Recorded scale"),
                                   (axes[1], np.log1p, "log1p scale")]:
        a, p = transform(actual), transform(predicted)
        maximum = max(float(a.max()), float(p.max()), 1.0)
        axis.scatter(a, p, s=13, alpha=.32, color=BLUE, edgecolors="none", rasterized=True)
        axis.plot([0, maximum], [0, maximum], color=ORANGE, linestyle="--", linewidth=1.4,
                  label="Perfect prediction")
        axis.set_xlabel("Actual recorded rejection" if title == "Recorded scale" else "log1p(actual)")
        axis.set_ylabel("Predicted recorded rejection" if title == "Recorded scale" else "log1p(predicted)")
        axis.set_title(title, fontsize=11)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{LABELS[chosen]} actual and predicted rejection on the test set", fontsize=13, fontweight="bold")
    save(fig, "actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.5), layout="constrained")
    ax.barh(positions - .18, cv.loc[names, "train_MAE"], height=.33, color=GRAY, label="Training folds")
    ax.barh(positions + .18, cv.loc[names, "cv_MAE"], height=.33, color=BLUE, label="Unseen validation families")
    ax.set_yticks(positions, [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("MAE in recorded rejection units")
    ax.set_title("Training and validation errors", loc="left", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, .5))
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    save(fig, "training_validation_gap.png")
    return saved
