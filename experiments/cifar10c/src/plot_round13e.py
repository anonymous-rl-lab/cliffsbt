#!/usr/bin/env python3
"""Create the formal Round 13E result figure from the frozen summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("baseline", "random", "coverage", "hazard")
LABELS = ("Baseline", "Random", "Coverage", "Hazard")
COLORS = ("#68717a", "#4c78a8", "#2a9d8f", "#e07a3f")
SEEDS = (71, 83, 97)


def metric(summary: dict, arm: str, name: str) -> np.ndarray:
    return np.asarray([summary["results"][str(seed)][arm]["metrics"][name] for seed in SEEDS])


def add_arm_panel(ax: plt.Axes, summary: dict, name: str, title: str) -> None:
    values = [metric(summary, arm, name) for arm in ARMS]
    means = [float(row.mean()) for row in values]
    x = np.arange(len(ARMS))
    ax.bar(x, means, color=COLORS, width=0.68, alpha=0.92)
    offsets = (-0.11, 0.0, 0.11)
    for arm_index, row in enumerate(values):
        for offset, value in zip(offsets, row):
            ax.scatter(arm_index + offset, value, color="white", edgecolor="#1f2933", s=30, zorder=3, linewidth=0.8)
    for index, value in enumerate(means):
        ax.text(index, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Error (lower is better)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(0.42, 0.64)
    ax.grid(axis="y", color="#d9dee3", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    selections = json.loads(args.selections.read_text(encoding="utf-8"))
    stats = summary["selection_stats"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#7b8794",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)
    add_arm_panel(axes[0, 0], summary, "endpoint_error_mean", "A  Endpoint risk on 15 official corruption streams")
    add_arm_panel(axes[0, 1], summary, "risk_area_mean", "B  Risk area across five ordered severities")

    ax = axes[1, 0]
    selection_arms = ("random", "hazard", "coverage")
    y = np.arange(3)
    fragments = [stats[arm]["unique_fragments"] for arm in selection_arms]
    bars = ax.barh(y, fragments, color=(COLORS[1], COLORS[3], COLORS[2]), height=0.62)
    ax.set_yticks(y, ("Random", "Hazard", "Coverage"))
    ax.invert_yaxis()
    ax.set_xlabel("Unique corruption × class × first-severity fragments")
    ax.set_title("C  Selection breadth with dangerous-hit rate", loc="left", fontweight="bold")
    ax.set_xlim(0, 810)
    ax.grid(axis="x", color="#d9dee3", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, arm, count in zip(bars, selection_arms, fragments):
        hit = 100 * stats[arm]["hazard_hit_rate"]
        ax.text(count + 12, bar.get_y() + bar.get_height() / 2, f"{count} fragments  |  {hit:.1f}% hits", va="center", fontsize=9)
    hazard_scores = {}
    for arm in ("hazard", "coverage"):
        hazard_scores[arm] = float(np.mean([row["hazard_score"] for row in selections["selections"][arm]]))
    ax.text(0.02, -0.27, f"Mean boundary-pressure score: hazard {hazard_scores['hazard']:.2f}; coverage {hazard_scores['coverage']:.2f}", transform=ax.transAxes, fontsize=9, color="#38434d")

    ax = axes[1, 1]
    contrasts = summary["comparisons"]
    rows = (
        ("Endpoint error", "coverage_minus_hazard_endpoint_error_mean"),
        ("Risk area", "coverage_minus_hazard_risk_area_mean"),
        ("Crossing fraction", "coverage_minus_hazard_crossing_fraction"),
    )
    estimates = np.asarray([100 * contrasts[key]["estimate"] for _, key in rows])
    low = np.asarray([100 * contrasts[key]["ci95"][0] for _, key in rows])
    high = np.asarray([100 * contrasts[key]["ci95"][1] for _, key in rows])
    y = np.arange(len(rows))
    ax.axvline(0, color="#20262e", linewidth=1.1)
    ax.errorbar(estimates, y, xerr=np.vstack((estimates - low, high - estimates)), fmt="o", color="#b64e2d", ecolor="#b64e2d", elinewidth=2, capsize=4, markersize=7)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Coverage − hazard (percentage points; positive is worse)")
    ax.set_title("D  Pretarget paired seed-cluster contrasts", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9dee3", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for yi, estimate in zip(y, estimates):
        ax.text(estimate + 0.22, yi - 0.13, f"{estimate:+.2f}", fontsize=9, fontweight="bold", color="#8f381f")

    fig.suptitle(
        "Round 13E — broader dangerous support repaired baseline, but deep hazard concentration won the fixed budget",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "3 fresh training seeds · 8,000 sealed identities · 15 CIFAR-10-C corruptions · 5 severities · 1,000-image repair budget · formal decision: PARTIAL_OR_STOP (6/9)",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()

