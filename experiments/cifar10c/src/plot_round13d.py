#!/usr/bin/env python3
"""Create the publication-style Round 13D result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    corruptions = data["config"]["corruptions"]
    seeds = data["config"]["seeds"]
    cells = data["cells"]

    delta = np.array(
        [[next(c for c in cells if c["seed"] == seed and c["corruption"] == corr)["endpoint_risk_increase"] for corr in corruptions] for seed in seeds]
    )
    cliff = np.array(
        [[next(c for c in cells if c["seed"] == seed and c["corruption"] == corr)["cliff_level"] or 0 for corr in corruptions] for seed in seeds]
    )
    all_cross = np.all(cliff > 0, axis=0)
    any_cross = np.any(cliff > 0, axis=0)
    colors = np.where(all_cross, "#9E2A2B", np.where(any_cross, "#E09F3E", "#5B6770"))
    labels = [c.replace("_", " ") for c in corruptions]
    y = np.arange(len(labels))

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 7.2), gridspec_kw={"width_ratios": [1.45, 0.78]})
    means = delta.mean(axis=0)
    lower = means - delta.min(axis=0)
    upper = delta.max(axis=0) - means
    ax1.barh(y, means, xerr=np.vstack([lower, upper]), color=colors, alpha=0.92, capsize=2.5, edgecolor="white", linewidth=0.6)
    ax1.axvline(0.15, color="#1D3557", linestyle="--", linewidth=1.5)
    ax1.text(0.153, -0.75, "Frozen headroom = 0.15", color="#1D3557", fontsize=8, va="center")
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Endpoint error increase")
    ax1.set_title("A  Risk growth and operational crossing")
    ax1.grid(axis="x", alpha=0.18)
    ax1.spines[["top", "right", "left"]].set_visible(False)

    display = cliff.T
    cmap = ListedColormap(["#E5E7EB", "#F6BD60", "#F28482", "#D95D39", "#9E2A2B", "#6A1B1A"])
    image = ax2.imshow(display, aspect="auto", cmap=cmap, vmin=0, vmax=5)
    ax2.set_xticks(np.arange(len(seeds)), [f"seed {seed}" for seed in seeds])
    ax2.set_yticks(y, labels)
    ax2.set_title("B  First severity exhausting headroom")
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = int(display[row, col])
            ax2.text(col, row, "—" if value == 0 else str(value), ha="center", va="center", color="white" if value >= 3 else "#111827", fontsize=8, fontweight="bold")
    ax2.tick_params(length=0)
    ax2.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.04, ticks=range(6))
    colorbar.ax.set_yticklabels(["no cliff", "1", "2", "3", "4", "5"])
    colorbar.set_label("Severity level")

    fig.suptitle("Round 13D | Official CIFAR-10-C paired boundary transport", fontsize=14, fontweight="bold", x=0.51)
    fig.text(0.51, 0.015, "Bars show training-seed mean and range. Red: all seeds cross; orange: mixed; gray: no seed crosses.", ha="center", fontsize=8.5, color="#374151")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

