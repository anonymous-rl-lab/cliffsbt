#!/usr/bin/env python3
"""Posttarget visualization for the frozen Round 11C results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cliff_round11c_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAG = "round11c_official_source_flux"


def main() -> None:
    result = ROOT / "results" / TAG
    paths = pd.read_csv(result / "path_summary.csv")
    paired = pd.read_csv(result / "paired_effects.csv")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    for index, (_, row) in enumerate(paired.iterrows()):
        color = "#2878b5" if row["path"] == "noise" else "#d95f02"
        axes[0].plot(
            [0, 1],
            [row["end_risk_baseline"], row["end_risk_aware"]],
            color=color,
            alpha=0.55,
            marker="o",
        )
    axes[0].set_xticks([0, 1], ["Baseline", "Cliff-aware"])
    axes[0].set_ylabel("Endpoint error risk")
    axes[0].set_title("Same stream, paired models")

    baseline = paths[paths["regime"] == "baseline"].reset_index(drop=True)
    x = np.arange(len(baseline))
    axes[1].scatter(
        baseline["first_crossing_entropy"],
        baseline["incident_persistence"],
        c=["#2878b5" if p == "noise" else "#d95f02" for p in baseline["path"]],
        s=45,
        alpha=0.85,
    )
    axes[1].axvline(0.65, color="black", ls="--", lw=1, alpha=0.6)
    axes[1].axhline(0.85, color="black", ls="--", lw=1, alpha=0.6)
    axes[1].set(
        xlabel="Normalized first-crossing entropy",
        ylabel="Endpoint persistence",
        title="Distributed and persistent flux",
        xlim=(0.6, 1.0),
        ylim=(0.8, 1.01),
    )

    values = summary["checks"]["values"]
    estimates = [
        values["mean_cliff_aware_end_risk_reduction"],
        values["mean_cliff_aware_incident_crossing_reduction"],
    ]
    intervals = [
        values["end_risk_reduction_cluster_ci95"],
        values["incident_crossing_reduction_cluster_ci95"],
    ]
    y = np.arange(2)
    errors = np.asarray(
        [[est - low for est, (low, _) in zip(estimates, intervals)],
         [high - est for est, (_, high) in zip(estimates, intervals)]]
    )
    axes[2].errorbar(estimates, y, xerr=errors, fmt="o", capsize=4, color="#3a923a")
    axes[2].axvline(0, color="black", lw=1)
    axes[2].set_yticks(y, ["Endpoint risk", "Incident crossing"])
    axes[2].set_xlabel("Cliff-aware reduction (95% seed-cluster CI)")
    axes[2].set_title("Training intervention reduces flux")
    axes[2].invert_yaxis()

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Round 11C — TorchSig 2.1.1 official-source execution", y=1.02)
    fig.tight_layout()
    output = ROOT / "figures" / TAG / f"{TAG}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
