#!/usr/bin/env python3
"""Build the publication figure for the frozen Round 12C result."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "round12c_coverage_vs_hazard_pilot"
OUTPUT = ROOT / "figures" / "round12c_coverage_vs_hazard_pilot"


def main() -> None:
    selections = pd.read_csv(RESULTS / "selection_summary.csv")
    effects = pd.read_csv(RESULTS / "seed_paired_effects.csv")
    contrasts = pd.read_csv(RESULTS / "seed_contrasts.csv")

    order = [
        "random_unstratified",
        "hazard_concentrated",
        "coverage_random",
        "coverage_hazard",
    ]
    labels = ["Random", "Hazard", "Coverage", "Coverage\n+ hazard"]
    colors = ["#9AA0A6", "#D55E00", "#0072B2", "#009E73"]
    selection_mean = selections.groupby("arm").mean(numeric_only=True).loc[order]
    effect_mean = effects.groupby("arm").mean(numeric_only=True).loc[order]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.5))
    x = np.arange(len(order))

    width = 0.35
    axes[0, 0].bar(
        x - width / 2,
        selection_mean["coverage_fraction"],
        width,
        color=colors,
        alpha=0.95,
        label="Deployment-cell coverage",
    )
    axes[0, 0].bar(
        x + width / 2,
        selection_mean["true_local_incident_precision"],
        width,
        color="white",
        edgecolor=colors,
        linewidth=1.4,
        hatch="///",
        label="Hazard-hit rate",
    )
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 0].set_title("a  Orthogonal selection manipulation", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[0, 1].bar(
        x,
        selection_mean["mean_local_hazard_score"],
        color=colors,
        width=0.62,
    )
    axes[0, 1].set_ylabel("Mean local hazard score")
    axes[0, 1].set_title("b  Hazard concentration is active", loc="left", fontweight="bold")

    rng = np.random.default_rng(20261260)
    for position, (arm, color) in enumerate(zip(order, colors)):
        values = effects.loc[effects["arm"] == arm, "mean_end_risk_reduction"].to_numpy()
        jitter = rng.normal(0, 0.035, len(values))
        axes[1, 0].scatter(
            np.full(len(values), position) + jitter,
            values,
            s=28,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        axes[1, 0].plot(
            [position - 0.22, position + 0.22],
            [effect_mean.loc[arm, "mean_end_risk_reduction"]] * 2,
            color=color,
            linewidth=3,
            solid_capstyle="round",
        )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_ylabel("Baseline minus repaired terminal risk")
    axes[1, 0].set_title("c  Fresh-stream repair across five seeds", loc="left", fontweight="bold")

    primary = contrasts["coverage_random_minus_hazard_concentrated"].to_numpy()
    official_ci = (0.036328125, 0.146484375)
    mean = float(primary.mean())
    axes[1, 1].axvline(0, color="#555555", linewidth=0.8)
    axes[1, 1].scatter(primary, np.arange(1, 6), s=30, color="#0072B2", zorder=3)
    axes[1, 1].errorbar(
        mean,
        0,
        xerr=np.asarray([[mean - official_ci[0]], [official_ci[1] - mean]]),
        fmt="o",
        color="#111111",
        ecolor="#111111",
        elinewidth=1.6,
        capsize=3,
        markersize=5,
    )
    axes[1, 1].set_yticks([0, 1, 2, 3, 4, 5])
    axes[1, 1].set_yticklabels(["Mean (95% CI)", "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5"])
    axes[1, 1].set_xlabel("Coverage minus hazard: terminal-risk reduction")
    axes[1, 1].set_title("d  Pretarget primary contrast", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
    axes[1, 1].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[1, 1].grid(axis="y", visible=False)

    figure.tight_layout(pad=1.4, w_pad=2.1, h_pad=2.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "round12c_coverage_vs_hazard_pilot.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT / "round12c_coverage_vs_hazard_pilot.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
