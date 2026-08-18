#!/usr/bin/env python3
"""Create the cross-run Round 4 audit ledger and summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    z = float(norm.ppf(1 - alpha / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def load_checks(tag: str) -> dict:
    path = ROOT / "results" / tag / "checks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def formal_row(tag: str) -> dict:
    directory = ROOT / "results" / tag
    panel = pd.read_csv(directory / "sealed_target_panel.csv")
    first = panel.iloc[0]
    cfg = json.loads((ROOT / "configs" / f"{tag}.json").read_text(encoding="utf-8"))
    reveal_n = int(cfg["target"]["reveal_samples_per_state"])
    minus_count = int(round(float(first["risk_minus_revealed"]) * reveal_n))
    plus_count = int(round(float(first["risk_plus_revealed"]) * reveal_n))
    minus_ci = wilson_interval(minus_count, reveal_n)
    plus_ci = wilson_interval(plus_count, reveal_n)
    boundary = float(first["boundary_risk_frozen"])
    gamma = float(first["requested_gamma"])
    midpoint = 0.5 * (float(first["risk_minus_revealed"]) + float(first["risk_plus_revealed"]))
    checks = load_checks(tag)
    return {
        "run": tag,
        "seed": cfg["master_seed"],
        "calibration_environments": len(pd.read_csv(directory / "calibration_environment_panel.csv")),
        "risk_linear_r2": checks["metrics"]["risk_linear_r2"],
        "relevant_score_linear_r2": checks["metrics"]["rich_relevant_score_linear_r2"],
        "null_ratio": checks["metrics"]["rich_null_ratio"],
        "theory_empirical_mae": checks["metrics"]["rich_theory_empirical_mae"],
        "theory_empirical_spearman": checks["metrics"]["rich_theory_empirical_spearman"],
        "risk_minus": float(first["risk_minus_revealed"]),
        "risk_minus_ci_low": minus_ci[0],
        "risk_minus_ci_high": minus_ci[1],
        "risk_plus": float(first["risk_plus_revealed"]),
        "risk_plus_ci_low": plus_ci[0],
        "risk_plus_ci_high": plus_ci[1],
        "boundary_risk_frozen": boundary,
        "gamma": gamma,
        "lower_cutoff": boundary - gamma,
        "higher_cutoff": boundary + gamma,
        "actual_half_gap": float(first["actual_half_gap"]),
        "midpoint_drift": midpoint - boundary,
        "relative_point_gate": bool(checks["metrics"]["rich_safe_cliff_realized"]),
        "overall_all_gates": bool(checks["all_passed"]),
    }


def plot_formal(rows: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colors = {"formal_identified": "#3568b8", "formal_identified_v2": "#c74c4c"}
    for tag in rows["run"]:
        panel = pd.read_csv(ROOT / "results" / tag / "sealed_target_panel.csv")
        axes[0].plot(
            panel["batch_size"], panel["theoretical_accuracy_frozen_gamma"],
            "--o", color=colors[tag], label=f"{tag}: theory"
        )
        axes[0].plot(
            panel["batch_size"], panel["operational_accuracy"],
            "-s", color=colors[tag], label=f"{tag}: frozen auditor"
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0.75, 1.01)
    axes[0].set_xlabel("Unlabelled target batch size")
    axes[0].set_ylabel("Balanced state accuracy")
    axes[0].set_title("Identifiability curve replicates")
    axes[0].legend(fontsize=7)

    x = np.arange(len(rows))
    width = 0.22
    axes[1].bar(x - width, rows["risk_minus"], width, label="revealed lower-risk state", color="#4f8a5b")
    axes[1].bar(x, rows["boundary_risk_frozen"], width, label="frozen boundary", color="#777777")
    axes[1].bar(x + width, rows["risk_plus"], width, label="revealed higher-risk state", color="#b84b4b")
    for i, row in rows.iterrows():
        axes[1].hlines(row["lower_cutoff"], i - 0.36, i - 0.08, colors="#1f6b2d", linestyles="--")
        axes[1].hlines(row["higher_cutoff"], i + 0.08, i + 0.36, colors="#8e2020", linestyles="--")
    axes[1].set_xticks(x, ["formal v1", "formal v2"])
    axes[1].set_ylabel("Deployment error")
    axes[1].set_title("Relative boundary gate abstains")
    axes[1].legend(fontsize=7)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "results" / "round4_summary"
    figure_dir = ROOT / "figures" / "round4_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([formal_row("formal_identified"), formal_row("formal_identified_v2")])
    rows.to_csv(output_dir / "formal_replication_ledger.csv", index=False)
    summary = {
        "formal_runs": rows.to_dict(orient="records"),
        "interpretation": {
            "identified_bridge": "replicated",
            "relative_risk_interval_gate": "abstain",
            "absolute_safety_claim": "not tested; no external TorchSig harm threshold",
            "next_hinge": "risk midpoint drift from curvature and boundary uncertainty",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_formal(rows, figure_dir / "formal_replication_summary.png")


if __name__ == "__main__":
    main()
