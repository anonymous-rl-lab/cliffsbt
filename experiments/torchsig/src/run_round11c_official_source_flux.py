#!/usr/bin/env python3
"""Round 11C source-faithful confirmation of distributed boundary flux."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# This install must precede every import in the frozen project that imports
# torchsig.  We intentionally do not expose this as a standard package run.
from torchsig_official_source_numpy_runtime import (  # noqa: E402
    RUNTIME_KIND,
    TORCHSIG_SOURCE_COMMIT,
    TORCHSIG_TAG,
    TORCHSIG_WHEEL_SHA256,
    install,
)

install()

import torch  # type: ignore  # noqa: E402
import torchsig  # type: ignore  # noqa: E402
from run_round11_paired_sync import (  # noqa: E402
    analyze_path,
    paired_panel,
    serializable,
    sha256,
    train_model,
    write_json,
)
from run_round11b_distributed_flux import crossing_shape  # noqa: E402


def freeze(config_relative: str, output: Path) -> str:
    cfg = json.loads((ROOT / config_relative).read_text(encoding="utf-8"))
    names = [
        config_relative,
        cfg["source_round10_config"],
        "ROUND11C_PRETARGET_PROTOCOL.md",
        "src/run_round11c_official_source_flux.py",
        "src/torchsig_official_source_numpy_runtime.py",
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


def cluster_interval(
    values: pd.Series, draws: int, confidence: float, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    distribution = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(distribution, alpha)), float(
        np.quantile(distribution, 1.0 - alpha)
    )


def build_checks(frame: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame, str]:
    baseline = frame[frame["regime"] == "baseline"].copy()
    aware = frame[frame["regime"] == "cliff_aware"].copy()
    paired = baseline.merge(
        aware, on=["replicate_seed", "path"], suffixes=("_baseline", "_aware")
    )
    paired["end_risk_reduction"] = (
        paired["end_risk_baseline"] - paired["end_risk_aware"]
    )
    paired["incident_crossing_reduction"] = (
        paired["incident_crossing_fraction_all_baseline"]
        - paired["incident_crossing_fraction_all_aware"]
    )
    seed_effects = paired.groupby("replicate_seed", as_index=False)[
        ["end_risk_reduction", "incident_crossing_reduction"]
    ].mean()
    boot = cfg["bootstrap"]
    end_ci = cluster_interval(
        seed_effects["end_risk_reduction"],
        int(boot["draws"]),
        float(boot["confidence"]),
        int(boot["seed"]),
    )
    incident_ci = cluster_interval(
        seed_effects["incident_crossing_reduction"],
        int(boot["draws"]),
        float(boot["confidence"]),
        int(boot["seed"]) + 1,
    )
    path_effects = paired.groupby("path")[
        ["end_risk_reduction", "incident_crossing_reduction"]
    ].median()
    gates = cfg["confirmation_gates"]
    values = {
        "maximum_flux_accounting_error": float(frame["maximum_flux_accounting_error"].max()),
        "baseline_relative_cliff_fraction": float(baseline["relative_cliff_crossed"].mean()),
        "median_incident_persistence": float(baseline["incident_persistence"].median()),
        "median_first_crossing_entropy": float(baseline["first_crossing_entropy"].median()),
        "median_three_window_incident_share": float(
            baseline["three_window_incident_share"].median()
        ),
        "velocity_coupling_pair_fraction": float(
            baseline["above_velocity_shuffle_q95"].mean()
        ),
        "true_boundary_specific_pair_fraction": float(
            baseline["above_random_boundary_q95"].mean()
        ),
        "mean_cliff_aware_end_risk_reduction": float(paired["end_risk_reduction"].mean()),
        "end_risk_reduction_cluster_ci95": list(end_ci),
        "mean_cliff_aware_incident_crossing_reduction": float(
            paired["incident_crossing_reduction"].mean()
        ),
        "incident_crossing_reduction_cluster_ci95": list(incident_ci),
        "minimum_pathwise_median_end_risk_reduction": float(
            path_effects["end_risk_reduction"].min()
        ),
        "minimum_pathwise_median_incident_crossing_reduction": float(
            path_effects["incident_crossing_reduction"].min()
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
        "end_risk_reduction_cluster_interval": end_ci[0]
        > gates["minimum_end_risk_reduction_cluster_ci_lower"],
        "cliff_aware_incident_crossing_reduction": values[
            "mean_cliff_aware_incident_crossing_reduction"
        ]
        >= gates["minimum_mean_cliff_aware_incident_crossing_reduction"],
        "incident_crossing_reduction_cluster_interval": incident_ci[0]
        > gates["minimum_incident_crossing_reduction_cluster_ci_lower"],
        "both_paths_improve": min(
            values["minimum_pathwise_median_end_risk_reduction"],
            values["minimum_pathwise_median_incident_crossing_reduction"],
        )
        > gates["minimum_pathwise_median_reduction"],
    }
    passed = int(sum(checks.values()))
    decision = (
        "OFFICIAL_SOURCE_NUMPY_CONFIRMATION_PASS_PACKAGE_RUNTIME_REQUIRED"
        if passed == len(checks)
        else "OFFICIAL_SOURCE_NUMPY_CONFIRMATION_STOP"
    )
    result = {"values": values, "checks": checks, "passed": passed, "total": len(checks)}
    return result, paired, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round11c_official_source_flux.json")
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
            f"pretarget digest mismatch or missing: observed={digest} "
            f"expected={args.expected_pretarget_sha}"
        )
    source_cfg = json.loads(
        (ROOT / probe["source_round10_config"]).read_text(encoding="utf-8")
    )
    source_cfg = copy.deepcopy(source_cfg)
    source_cfg["training"].update(probe["training_override"])
    signal_cfg = source_cfg["signal"]
    resample_rate = (
        float(signal_cfg["sample_rate"]) / float(signal_cfg["bandwidth"]) / 4.0
    )
    if not np.isclose(resample_rate, 1.0, rtol=0.0, atol=1e-15):
        raise SystemExit("frozen signal config is outside the source-runtime scope")
    panels, panel_hash = paired_panel(source_cfg, probe)
    path_rows: list[dict] = []
    transition_rows: list[dict] = []
    sample_rows: list[dict] = []
    training_rows: list[dict] = []
    archive: dict[str, np.ndarray] = {}
    for replicate_seed in probe["replicate_seeds"]:
        for regime in probe["regimes"]:
            model, training, _ = train_model(source_cfg, regime, int(replicate_seed))
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
                    int(replicate_seed),
                    source_cfg,
                    probe,
                )
                summary.update(
                    crossing_shape(
                        margin,
                        int(source_cfg["deployment"]["pre_plateau_windows"]),
                        int(probe["paired_deployment"]["confirmation_windows"]),
                    )
                )
                path_rows.append(summary)
                transition_rows.extend(transitions)
                sample_rows.extend(samples)
                key = f"seed{replicate_seed}_{regime}_{path}"
                archive[f"{key}_probabilities"] = probabilities.astype(np.float32)
                archive[f"{key}_margins"] = margin.astype(np.float32)
    path_frame = pd.DataFrame(path_rows)
    checks, paired, decision = build_checks(path_frame, probe)
    output.mkdir(parents=True, exist_ok=True)
    path_frame.to_csv(output / "path_summary.csv", index=False)
    paired.to_csv(output / "paired_effects.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "transition_flux.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "sample_first_crossings.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "training_summary.csv", index=False)
    np.savez_compressed(output / "paired_margin_trajectories.npz", **archive)
    result = {
        "pretarget_manifest_digest": digest,
        "standard_torchsig_package_runtime": False,
        "runtime": {
            "kind": RUNTIME_KIND,
            "python": platform.python_version(),
            "torch_stub": getattr(torch, "__version__", "unknown"),
            "torchsig_exposed_version": getattr(torchsig, "__version__", "unknown"),
            "torchsig_tag": TORCHSIG_TAG,
            "torchsig_source_commit": TORCHSIG_SOURCE_COMMIT,
            "torchsig_wheel_sha256_for_future_confirmation": TORCHSIG_WHEEL_SHA256,
            "ideal_resample_rate": resample_rate,
        },
        "paired_panel_sha256": panel_hash,
        "models": len(training_rows),
        "model_path_pairs": len(path_frame),
        "checks": checks,
        "decision": decision,
        "claim_status": (
            "source-faithful signal-algorithm confirmation only; installed "
            "PyTorch/TorchSig package-runtime replay remains required"
        ),
    }
    write_json(output / "summary.json", result)
    print(json.dumps(serializable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
