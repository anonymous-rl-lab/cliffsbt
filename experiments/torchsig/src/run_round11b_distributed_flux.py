#!/usr/bin/env python3
"""Fresh-seed test of the distributed-flux replacement for synchronization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_round11_paired_sync import (
    ROOT,
    STANDARD_RUNTIME,
    analyze_path,
    confirmed_first_crossing,
    paired_panel,
    serializable,
    sha256,
    train_model,
    write_json,
)


def freeze(config_relative: str, output: Path) -> str:
    cfg = json.loads((ROOT / config_relative).read_text(encoding="utf-8"))
    names = [
        config_relative,
        cfg["source_round10_config"],
        "ROUND11B_PRETARGET_PROTOCOL.md",
        "src/run_round11b_distributed_flux.py",
        "src/run_round11_paired_sync.py",
        "src/torchsig_numpy_compat.py",
        "src/run_round10_training_intervention.py",
        "src/run_pilot.py",
    ]
    content = "\n".join(f"{sha256(ROOT / name)}  {name}" for name in names) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "PRETARGET_RELEASE_SHA256.txt").write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    (output / "PRETARGET_MANIFEST_DIGEST.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


def crossing_shape(margin: np.ndarray, anchor: int, confirmation: int) -> dict:
    first = confirmed_first_crossing(margin, anchor, confirmation)
    incident_mask = first >= 0
    incident = first[incident_mask]
    counts = np.asarray(
        [np.sum(first == time) for time in range(anchor + 1, margin.shape[0])], dtype=float
    )
    positive = counts[counts > 0]
    if len(positive) > 1:
        weights = positive / positive.sum()
        entropy = float(-np.sum(weights * np.log(weights)) / np.log(len(counts)))
    else:
        entropy = 0.0
    three_peak = max(
        (counts[index : index + 3].sum() for index in range(max(len(counts) - 2, 1))),
        default=0.0,
    )
    return {
        "incident_persistence": float(np.mean(margin[-1, incident_mask] <= 0))
        if np.any(incident_mask)
        else 0.0,
        "first_crossing_entropy": entropy,
        "three_window_incident_share": float(three_peak / max(counts.sum(), 1.0)),
        "distinct_first_crossing_windows": int(np.sum(counts > 0)),
    }


def build_checks(frame: pd.DataFrame, gates: dict) -> tuple[dict, str]:
    baseline = frame[frame["regime"] == "baseline"].copy()
    aware = frame[frame["regime"] == "cliff_aware"].copy()
    paired = baseline.merge(aware, on=["replicate_seed", "path"], suffixes=("_baseline", "_aware"))
    values = {
        "maximum_flux_accounting_error": float(frame["maximum_flux_accounting_error"].max()),
        "baseline_relative_cliff_fraction": float(baseline["relative_cliff_crossed"].mean()),
        "median_incident_persistence": float(baseline["incident_persistence"].median()),
        "median_first_crossing_entropy": float(baseline["first_crossing_entropy"].median()),
        "median_three_window_incident_share": float(
            baseline["three_window_incident_share"].median()
        ),
        "velocity_coupling_pair_fraction": float(baseline["above_velocity_shuffle_q95"].mean()),
        "true_boundary_specific_pair_fraction": float(baseline["above_random_boundary_q95"].mean()),
        "mean_cliff_aware_end_risk_reduction": float(
            (paired["end_risk_baseline"] - paired["end_risk_aware"]).mean()
        ),
        "mean_cliff_aware_incident_crossing_reduction": float(
            (
                paired["incident_crossing_fraction_all_baseline"]
                - paired["incident_crossing_fraction_all_aware"]
            ).mean()
        ),
    }
    checks = {
        "flux_accounting": values["maximum_flux_accounting_error"]
        <= gates["maximum_flux_accounting_error"],
        "baseline_operational_cliff": values["baseline_relative_cliff_fraction"]
        >= gates["minimum_baseline_relative_cliff_fraction"],
        "incident_persistence": values["median_incident_persistence"]
        >= gates["minimum_median_incident_persistence"],
        "temporally_distributed_crossing": values["median_first_crossing_entropy"]
        >= gates["minimum_median_first_crossing_entropy"],
        "no_dominant_three_window_pulse": values["median_three_window_incident_share"]
        <= gates["maximum_median_three_window_incident_share"],
        "no_required_extra_velocity_coupling": values["velocity_coupling_pair_fraction"]
        <= gates["maximum_velocity_coupling_pair_fraction"],
        "true_boundary_specificity": values["true_boundary_specific_pair_fraction"]
        >= gates["minimum_true_boundary_specific_pair_fraction"],
        "cliff_aware_end_risk_reduction": values["mean_cliff_aware_end_risk_reduction"]
        >= gates["minimum_mean_cliff_aware_end_risk_reduction"],
        "cliff_aware_incident_crossing_reduction": values[
            "mean_cliff_aware_incident_crossing_reduction"
        ]
        >= gates["minimum_mean_cliff_aware_incident_crossing_reduction"],
    }
    decision = (
        "DISTRIBUTED_FLUX_COMPAT_PROBE_PASS_STANDARD_RUNTIME_REQUIRED"
        if all(checks.values())
        else "DISTRIBUTED_FLUX_PROBE_STOP"
    )
    return {"values": values, "checks": checks, "passed": sum(checks.values()), "total": len(checks)}, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round11b_distributed_flux_probe.json")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--expected-pretarget-sha")
    args = parser.parse_args()
    probe = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / "results" / probe["output_tag"]
    digest = freeze(args.config, output)
    if args.freeze_only:
        print(digest)
        return
    if args.expected_pretarget_sha is None or digest != args.expected_pretarget_sha:
        raise SystemExit(
            f"pretarget digest mismatch or missing: observed={digest} expected={args.expected_pretarget_sha}"
        )
    cfg = json.loads((ROOT / probe["source_round10_config"]).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["training"].update(probe["training_override"])
    panels, panel_hash = paired_panel(cfg, probe)
    path_rows = []
    transition_rows = []
    sample_rows = []
    training_rows = []
    for seed in probe["replicate_seeds"]:
        for regime in probe["regimes"]:
            model, training, _ = train_model(cfg, regime, int(seed))
            training_rows.append(training)
            for path, panel in panels.items():
                probabilities = np.stack(
                    [model.predict_proba(item) for item in panel["features"]], axis=0
                )
                summary, transitions, samples, margin = analyze_path(
                    probabilities,
                    panel["labels"],
                    regime,
                    path,
                    int(seed),
                    cfg,
                    probe,
                )
                summary.update(
                    crossing_shape(
                        margin,
                        int(cfg["deployment"]["pre_plateau_windows"]),
                        int(probe["paired_deployment"]["confirmation_windows"]),
                    )
                )
                path_rows.append(summary)
                transition_rows.extend(transitions)
                sample_rows.extend(samples)
    path_frame = pd.DataFrame(path_rows)
    checks, decision = build_checks(path_frame, probe["fresh_probe_gates"])
    output.mkdir(parents=True, exist_ok=True)
    path_frame.to_csv(output / "path_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "transition_flux.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "sample_first_crossings.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "training_summary.csv", index=False)
    result = {
        "pretarget_manifest_digest": digest,
        "standard_torchsig_runtime": STANDARD_RUNTIME,
        "paired_panel_sha256": panel_hash,
        "models": len(training_rows),
        "model_path_pairs": len(path_frame),
        "checks": checks,
        "decision": decision,
        "claim_status": "fresh redesign probe only; standard TorchSig runtime required",
    }
    write_json(output / "summary.json", result)
    print(json.dumps(serializable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
