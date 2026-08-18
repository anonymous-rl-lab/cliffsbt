#!/usr/bin/env python3
"""Build the frozen quadratic v3 ledger and summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "formal_quadratic_v3"


def main() -> None:
    panel = pd.read_csv(RUN / "sealed_target_panel.csv")
    checks = json.loads((RUN / "checks.json").read_text(encoding="utf-8"))
    fit = json.loads((RUN / "frozen_estimands.json").read_text(encoding="utf-8"))["rich"]
    first = panel.iloc[0]
    output = RUN / "quadratic_v3_ledger.json"
    ledger = {
        "pretarget_gate": json.loads((RUN / "pretarget_freeze_gate.json").read_text(encoding="utf-8")),
        "final_checks": checks,
        "frozen_pair": {
            "tau": fit["risk_intercept"],
            "b": fit["b"],
            "H": fit["H"],
            "u_minus": fit["quadratic_pair"]["u_minus"],
            "u_plus": fit["quadratic_pair"]["u_plus"],
            "design_margin": fit["quadratic_pair"]["design_margin"],
            "certification_gamma": fit["quadratic_pair"]["certification_gamma"],
            "support_slack_min": fit["quadratic_pair"]["support_slack_min"],
        },
        "revealed": {
            "lower_risk_state": float(first["risk_minus_revealed"]),
            "higher_risk_state": float(first["risk_plus_revealed"]),
            "lower_risk_ci95": checks["metrics"]["risk_minus_ci95"],
            "higher_risk_ci95": checks["metrics"]["risk_plus_ci95"],
            "relative_point_gate": checks["metrics"]["rich_safe_cliff_realized"],
            "relative_ci_gate": checks["metrics"]["rich_safe_cliff_ci_realized"],
        },
        "semantics": "protocol-relative risk separation; not an absolute safety threshold",
        "decision": "relative point gate passes; frozen two-sided 95% interval gate abstains",
    }
    output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(panel["batch_size"], panel["theoretical_accuracy_frozen_gamma"], "--o", label="quadratic theory")
    axes[0].plot(panel["batch_size"], panel["operational_accuracy"], "-s", label="frozen auditor")
    axes[0].plot(panel["batch_size"], panel["oracle_linear_accuracy"], ":^", label="target-label oracle")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0.75, 1.01)
    axes[0].set_xlabel("Unlabelled target batch size")
    axes[0].set_ylabel("Balanced state accuracy")
    axes[0].set_title("Quadratic asymmetric pair")
    axes[0].legend(fontsize=8)

    tau = float(first["boundary_risk_frozen"])
    gamma = float(first["requested_gamma"])
    values = np.array([first["risk_minus_revealed"], first["risk_plus_revealed"]], dtype=float)
    ci = np.asarray([checks["metrics"]["risk_minus_ci95"], checks["metrics"]["risk_plus_ci95"]])
    errors = np.vstack([values - ci[:, 0], ci[:, 1] - values])
    axes[1].errorbar([0, 1], values, yerr=errors, fmt="o", capsize=6, markersize=8, color="#2f5597")
    axes[1].axhline(tau, color="#666666", label="frozen boundary")
    axes[1].axhline(tau - gamma, color="#2d7a3e", linestyle="--", label="lower cutoff")
    axes[1].axhline(tau + gamma, color="#a02b2b", linestyle="--", label="higher cutoff")
    axes[1].set_xticks([0, 1], ["lower-risk state", "higher-risk state"])
    axes[1].set_ylabel("Revealed deployment error")
    axes[1].set_title("Relative point gate passes; CI gate abstains")
    axes[1].legend(fontsize=8)
    figure_dir = ROOT / "figures" / "formal_quadratic_v3"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "quadratic_v3_summary.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
