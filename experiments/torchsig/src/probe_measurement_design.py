#!/usr/bin/env python3
"""Enumerate measurement designs on the frozen quadratic v3 calibration fit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from measurement_design import evaluate_subsets, frontier
from run_pilot import optimize_supporting_pair


ROOT = Path(__file__).resolve().parents[1]


def serializable_row(row: dict) -> dict:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in row.items()
    }


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "measurement_design_probe.json").read_text())
    source = ROOT / "results" / "formal_quadratic_v3"
    fitted = json.loads((source / "frozen_estimands.json").read_text())["rich"]
    rows = evaluate_subsets(fitted, cfg, optimize_supporting_pair)
    budgets = [int(value) for value in cfg["measurement_design"]["budgets"]]
    chosen = frontier(rows, budgets, cfg)

    output = ROOT / "results" / "measurement_design_probe"
    figures = ROOT / "figures" / "measurement_design_probe"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([serializable_row(row) for row in rows]).to_csv(
        output / "all_subsets.csv", index=False
    )
    pd.DataFrame([serializable_row(row) for row in chosen]).to_csv(
        output / "frontier.csv", index=False
    )
    summary = {
        "source_run": "formal_quadratic_v3",
        "semantics": "calibration-only exploratory measurement-design probe",
        "number_of_group_subsets": len(rows),
        "number_optimizer_feasible": sum(bool(row["optimizer_feasible"]) for row in rows),
        "frontier": [serializable_row(row) for row in chosen],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = pd.DataFrame([serializable_row(row) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    feasible = table[table["optimizer_feasible"]]
    axes[0].scatter(
        feasible["cost"], feasible["pair_dQd_per_sample"],
        c=feasible["risk_null_ratio"], cmap="viridis_r", alpha=0.75
    )
    risk = pd.DataFrame([row for row in chosen if row["objective"] == "risk_directed"])
    trace = pd.DataFrame([row for row in chosen if row["objective"] == "trace"])
    axes[0].plot(risk["cost"], risk["pair_dQd_per_sample"], "-o", label="risk-directed")
    axes[0].plot(trace["cost"], trace["pair_dQd_per_sample"], "--s", label="trace(Q)")
    axes[0].set_xlabel("Measurement cost (scalar dimensions)")
    axes[0].set_ylabel("Candidate worst-pair $D_Q^2$")
    axes[0].set_title("Calibration-only design frontier")
    axes[0].legend(fontsize=8)

    axes[1].plot(risk["cost"], risk["predicted_n_target"], "-o", label="risk-directed")
    axes[1].plot(trace["cost"], trace["predicted_n_target"], "--s", label="trace(Q)")
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Measurement cost (scalar dimensions)")
    axes[1].set_ylabel("Predicted batch size for target accuracy")
    axes[1].set_title(f"Predicted n for accuracy {cfg['measurement_design']['target_accuracy']:.2f}")
    axes[1].legend(fontsize=8)
    fig.savefig(figures / "measurement_design_probe.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()

