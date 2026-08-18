#!/usr/bin/env python3
"""Round 11A paired-sample synchronized decision-boundary crossing smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cliff_round11_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch  # type: ignore
    import torchsig  # type: ignore
    STANDARD_RUNTIME = True
except ImportError:
    from torchsig_numpy_compat import install

    install()
    import torch  # type: ignore
    import torchsig  # type: ignore

    STANDARD_RUNTIME = False

from probe_cliff_early_warning import trajectory_scalars
from run_pilot import FEATURE_NAMES, extract_features, generate_iq, stable_seed
from run_round10_training_intervention import path_specs, train_model


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretarget_files(config_relative: str, source_config_relative: str) -> list[str]:
    return [
        config_relative,
        source_config_relative,
        "ROUND11A_PRETARGET_PROTOCOL.md",
        "src/run_round11_paired_sync.py",
        "src/torchsig_numpy_compat.py",
        "src/run_round10_training_intervention.py",
        "src/run_pilot.py",
        "src/probe_cliff_early_warning.py",
    ]


def freeze_pretarget(config_relative: str, output_dir: Path) -> str:
    cfg = read_json(ROOT / config_relative)
    names = pretarget_files(config_relative, cfg["source_round10_config"])
    content = "\n".join(f"{sha256(ROOT / name)}  {name}" for name in names) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    release = output_dir / "PRETARGET_RELEASE_SHA256.txt"
    release.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    (output_dir / "PRETARGET_MANIFEST_DIGEST.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


def true_margin(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = np.arange(len(labels))
    true_probability = probabilities[rows, labels]
    other = probabilities.copy()
    other[rows, labels] = -np.inf
    return true_probability - np.max(other, axis=1)


def paired_panel(cfg: dict, smoke: dict) -> tuple[dict[str, dict], str]:
    count = int(smoke["paired_deployment"]["samples_per_path"])
    classes = cfg["classes"]
    label_rng = np.random.default_rng(stable_seed(smoke["master_seed"], "paired_labels"))
    labels = np.arange(count, dtype=int) % len(classes)
    label_rng.shuffle(labels)
    center = np.asarray(cfg["calibration"]["theta_center"], dtype=float)
    panels: dict[str, dict] = {}
    digest = hashlib.sha256()
    for path in path_specs(cfg):
        direction = path["direction"] / max(np.linalg.norm(path["direction"]), 1e-12)
        scalars = trajectory_scalars(
            path["start_scalar"], path["end_scalar"], cfg["deployment"]
        )
        features = np.empty((len(scalars), count, len(FEATURE_NAMES)), dtype=np.float64)
        for time_index, scalar in enumerate(scalars):
            theta = center + float(scalar) * direction
            for sample_index, label in enumerate(labels):
                # Resetting the same sample seed at every time preserves symbols,
                # phase-noise variates, and AWGN variates across the path.
                rng = np.random.default_rng(
                    stable_seed(
                        smoke["master_seed"], "paired_latent", path["name"], sample_index
                    )
                )
                iq = generate_iq(classes[int(label)], theta, rng, cfg["signal"])
                features[time_index, sample_index] = extract_features(iq)
        panels[path["name"]] = {
            "labels": labels.copy(),
            "features": features,
            "scalars": scalars,
        }
        digest.update(path["name"].encode("utf-8"))
        digest.update(np.ascontiguousarray(labels).tobytes())
        digest.update(np.ascontiguousarray(features).tobytes())
        digest.update(np.ascontiguousarray(scalars).tobytes())
    return panels, digest.hexdigest()


def confirmed_first_crossing(
    margin: np.ndarray, anchor: int, confirmation: int
) -> np.ndarray:
    sample_count = margin.shape[1]
    result = np.full(sample_count, -1, dtype=int)
    at_risk = margin[anchor] > 0
    for sample in np.flatnonzero(at_risk):
        for time_index in range(anchor + 1, margin.shape[0] - confirmation + 1):
            if bool(np.all(margin[time_index : time_index + confirmation, sample] <= 0)):
                result[sample] = time_index
                break
    return result


def classwise_shuffle(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = values.copy()
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        shuffled[indices] = values[rng.permutation(indices)]
    return shuffled


def velocity_shuffle_null(
    margin: np.ndarray,
    labels: np.ndarray,
    anchor: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=float)
    count = margin.shape[1]
    for draw in range(draws):
        peaks = []
        for time_index in range(anchor, margin.shape[0] - 1):
            current = margin[time_index]
            velocity = margin[time_index + 1] - current
            shuffled = classwise_shuffle(velocity, labels, rng)
            peaks.append(np.mean((current > 0) & (current + shuffled <= 0)))
        result[draw] = max(peaks, default=0.0)
    return result


def random_boundary_null(
    probabilities: np.ndarray,
    true_margin_values: np.ndarray,
    anchor: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=float)
    initial_wrong = float(np.mean(true_margin_values[anchor] <= 0))
    for draw in range(draws):
        normal = rng.normal(size=probabilities.shape[2])
        normal -= normal.mean()
        normal /= max(np.linalg.norm(normal), 1e-12)
        score = probabilities @ normal
        threshold = float(np.quantile(score[anchor], initial_wrong))
        pseudo_margin = score - threshold
        peaks = [
            np.mean((pseudo_margin[t] > 0) & (pseudo_margin[t + 1] <= 0))
            for t in range(anchor, len(pseudo_margin) - 1)
        ]
        result[draw] = max(peaks, default=0.0)
    return result


def analyze_path(
    probabilities: np.ndarray,
    labels: np.ndarray,
    regime: str,
    path: str,
    replicate_seed: int,
    cfg: dict,
    smoke: dict,
) -> tuple[dict, list[dict], list[dict], np.ndarray]:
    margin = np.vstack([true_margin(item, labels) for item in probabilities])
    risk = np.mean(margin <= 0, axis=1)
    anchor = int(cfg["deployment"]["pre_plateau_windows"])
    confirmation = int(smoke["paired_deployment"]["confirmation_windows"])
    first = confirmed_first_crossing(margin, anchor, confirmation)
    transition_rows: list[dict] = []
    maximum_error = 0.0
    forward_peak = 0.0
    for time_index in range(anchor, len(risk) - 1):
        forward = int(np.sum((margin[time_index] > 0) & (margin[time_index + 1] <= 0)))
        recovery = int(np.sum((margin[time_index] <= 0) & (margin[time_index + 1] > 0)))
        observed_delta = float(risk[time_index + 1] - risk[time_index])
        accounted_delta = float((forward - recovery) / len(labels))
        error = abs(observed_delta - accounted_delta)
        maximum_error = max(maximum_error, error)
        forward_peak = max(forward_peak, forward / len(labels))
        transition_rows.append(
            {
                "replicate_seed": replicate_seed,
                "regime": regime,
                "path": path,
                "time_index": time_index,
                "risk": risk[time_index],
                "next_risk": risk[time_index + 1],
                "risk_delta": observed_delta,
                "forward_crossings": forward,
                "recoveries": recovery,
                "accounted_risk_delta": accounted_delta,
                "accounting_error": error,
            }
        )
    at_risk = margin[anchor] > 0
    incident = first >= 0
    incident_counts = np.asarray(
        [np.sum(first == time_index) for time_index in range(anchor + 1, len(risk))],
        dtype=float,
    )
    confirmed_peak = float(incident_counts.max() / len(labels)) if len(incident_counts) else 0.0
    relative_boundary = float(risk[anchor] + smoke["paired_deployment"]["relative_cliff_margin"])
    above = risk >= relative_boundary
    cliff_time = None
    for time_index in range(anchor, len(risk) - confirmation + 1):
        if bool(np.all(above[time_index : time_index + confirmation])):
            cliff_time = time_index
            break
    velocity_null = velocity_shuffle_null(
        margin,
        labels,
        anchor,
        int(smoke["paired_deployment"]["velocity_shuffle_draws"]),
        stable_seed(smoke["master_seed"], "velocity_null", replicate_seed, regime, path),
    )
    boundary_null = random_boundary_null(
        probabilities,
        margin,
        anchor,
        int(smoke["paired_deployment"]["random_boundary_draws"]),
        stable_seed(smoke["master_seed"], "boundary_null", replicate_seed, regime, path),
    )
    sample_rows = [
        {
            "replicate_seed": replicate_seed,
            "regime": regime,
            "path": path,
            "sample_index": sample,
            "class_index": int(labels[sample]),
            "margin_at_anchor": float(margin[anchor, sample]),
            "at_risk": bool(at_risk[sample]),
            "confirmed_first_crossing_time": int(first[sample]),
        }
        for sample in range(len(labels))
    ]
    summary = {
        "replicate_seed": replicate_seed,
        "regime": regime,
        "path": path,
        "panel_size": len(labels),
        "anchor_time": anchor,
        "anchor_risk": float(risk[anchor]),
        "end_risk": float(risk[-1]),
        "end_risk_increase": float(risk[-1] - risk[anchor]),
        "relative_boundary": relative_boundary,
        "relative_cliff_crossed": cliff_time is not None,
        "relative_cliff_time": cliff_time,
        "at_risk_count": int(at_risk.sum()),
        "incident_crossing_count": int(incident.sum()),
        "incident_crossing_fraction_all": float(incident.mean()),
        "incident_crossing_fraction_at_risk": float(incident.sum() / max(at_risk.sum(), 1)),
        "forward_crossing_peak_fraction": forward_peak,
        "confirmed_first_crossing_peak_fraction": confirmed_peak,
        "maximum_flux_accounting_error": maximum_error,
        "velocity_shuffle_median": float(np.median(velocity_null)),
        "velocity_shuffle_q95": float(np.quantile(velocity_null, 0.95)),
        "above_velocity_shuffle_q95": bool(forward_peak > np.quantile(velocity_null, 0.95)),
        "random_boundary_median": float(np.median(boundary_null)),
        "random_boundary_q95": float(np.quantile(boundary_null, 0.95)),
        "above_random_boundary_q95": bool(forward_peak > np.quantile(boundary_null, 0.95)),
    }
    return summary, transition_rows, sample_rows, margin


def build_checks(path_frame: pd.DataFrame, smoke: dict) -> tuple[dict, str]:
    gates = smoke["smoke_gates"]
    baseline = path_frame[path_frame["regime"] == "baseline"].copy()
    aware = path_frame[path_frame["regime"] == "cliff_aware"].copy()
    paired = baseline.merge(aware, on=["replicate_seed", "path"], suffixes=("_baseline", "_aware"))
    values = {
        "maximum_flux_accounting_error": float(path_frame["maximum_flux_accounting_error"].max()),
        "minimum_baseline_end_risk_increase": float(baseline["end_risk_increase"].min()),
        "minimum_baseline_incident_crossing_fraction": float(
            baseline["incident_crossing_fraction_all"].min()
        ),
        "paths_above_velocity_shuffle_q95": int(baseline["above_velocity_shuffle_q95"].sum()),
        "paths_above_random_boundary_q95": int(baseline["above_random_boundary_q95"].sum()),
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
        <= float(gates["maximum_flux_accounting_error"]),
        "baseline_risk_motion": values["minimum_baseline_end_risk_increase"]
        >= float(gates["minimum_baseline_end_risk_increase"]),
        "baseline_incident_crossing": values["minimum_baseline_incident_crossing_fraction"]
        >= float(gates["minimum_baseline_incident_crossing_fraction"]),
        "velocity_matching_specificity": values["paths_above_velocity_shuffle_q95"]
        >= int(gates["minimum_paths_above_velocity_shuffle_q95"]),
        "model_boundary_specificity": values["paths_above_random_boundary_q95"]
        >= int(gates["minimum_paths_above_random_boundary_q95"]),
        "cliff_aware_end_risk_reduction": values["mean_cliff_aware_end_risk_reduction"]
        >= float(gates["minimum_cliff_aware_end_risk_reduction"]),
        "cliff_aware_incident_crossing_reduction": values[
            "mean_cliff_aware_incident_crossing_reduction"
        ]
        >= float(gates["minimum_cliff_aware_incident_crossing_reduction"]),
    }
    decision = "ADVANCE_TO_PILOT" if all(checks.values()) else "SMOKE_STOP_REDESIGN"
    return {"values": values, "checks": checks, "passed": sum(checks.values()), "total": len(checks)}, decision


def make_figure(path_frame: pd.DataFrame, transitions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for (regime, path), frame in transitions.groupby(["regime", "path"], sort=False):
        frame = frame.sort_values("time_index")
        time = np.r_[frame["time_index"].to_numpy(), frame["time_index"].to_numpy()[-1] + 1]
        risk = np.r_[frame["risk"].to_numpy(), frame["next_risk"].to_numpy()[-1]]
        axes[0].plot(time, risk, marker="o", ms=3, label=f"{regime}: {path}")
        axes[1].plot(
            frame["time_index"] + 1,
            frame["forward_crossings"] / path_frame["panel_size"].iloc[0],
            marker="o",
            ms=3,
            label=f"{regime}: {path}",
        )
    axes[0].set(title="Paired risk trajectories", xlabel="window", ylabel="error risk")
    axes[1].set(title="Forward boundary-crossing flux", xlabel="window", ylabel="fraction")
    baseline = path_frame[path_frame["regime"] == "baseline"]
    x = np.arange(len(baseline))
    axes[2].bar(x - 0.25, baseline["forward_crossing_peak_fraction"], width=0.25, label="true")
    axes[2].bar(x, baseline["velocity_shuffle_q95"], width=0.25, label="velocity q95")
    axes[2].bar(x + 0.25, baseline["random_boundary_q95"], width=0.25, label="boundary q95")
    axes[2].set_xticks(x, baseline["path"], rotation=20)
    axes[2].set(title="Frozen placebo comparison", ylabel="peak crossing fraction")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round11_paired_sync_smoke.json")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--expected-pretarget-sha")
    args = parser.parse_args()
    smoke = read_json(ROOT / args.config)
    output = ROOT / "results" / smoke["output_tag"]
    digest = freeze_pretarget(args.config, output)
    if args.freeze_only:
        print(digest)
        return
    if args.expected_pretarget_sha is None or digest != args.expected_pretarget_sha:
        raise SystemExit(
            f"pretarget digest mismatch or missing: observed={digest} expected={args.expected_pretarget_sha}"
        )

    cfg = read_json(ROOT / smoke["source_round10_config"])
    cfg = copy.deepcopy(cfg)
    cfg["training"].update(smoke["training_override"])
    panels, panel_digest = paired_panel(cfg, smoke)
    path_rows: list[dict] = []
    transition_rows: list[dict] = []
    sample_rows: list[dict] = []
    archive: dict[str, np.ndarray] = {}
    training_rows = []
    for replicate_seed in smoke["replicate_seeds"]:
        for regime in smoke["regimes"]:
            model, training_summary, _ = train_model(cfg, regime, int(replicate_seed))
            training_rows.append(training_summary)
            for path, panel in panels.items():
                probabilities = np.stack(
                    [model.predict_proba(features) for features in panel["features"]], axis=0
                )
                summary, transitions, samples, margin = analyze_path(
                    probabilities,
                    panel["labels"],
                    regime,
                    path,
                    int(replicate_seed),
                    cfg,
                    smoke,
                )
                path_rows.append(summary)
                transition_rows.extend(transitions)
                sample_rows.extend(samples)
                key = f"seed{replicate_seed}_{regime}_{path}"
                archive[f"{key}_probabilities"] = probabilities.astype(np.float32)
                archive[f"{key}_margins"] = margin.astype(np.float32)
                archive[f"{key}_labels"] = panel["labels"].astype(np.int16)
                archive[f"{key}_scalars"] = panel["scalars"].astype(np.float32)
    path_frame = pd.DataFrame(path_rows)
    transition_frame = pd.DataFrame(transition_rows)
    sample_frame = pd.DataFrame(sample_rows)
    checks, decision = build_checks(path_frame, smoke)
    output.mkdir(parents=True, exist_ok=True)
    path_frame.to_csv(output / "path_summary.csv", index=False)
    transition_frame.to_csv(output / "transition_flux.csv", index=False)
    sample_frame.to_csv(output / "sample_first_crossings.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "training_summary.csv", index=False)
    np.savez_compressed(output / "paired_margin_trajectories.npz", **archive)
    summary = {
        "pretarget_manifest_digest": digest,
        "standard_torchsig_runtime": STANDARD_RUNTIME,
        "runtime": {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torchsig": getattr(torchsig, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "paired_panel_sha256": panel_digest,
        "models": len(training_rows),
        "paths": len(path_frame),
        "checks": checks,
        "decision": decision,
        "claim_status": "smoke-only; standard TorchSig confirmation required",
    }
    write_json(output / "summary.json", summary)
    make_figure(
        path_frame,
        transition_frame,
        ROOT / "figures" / smoke["output_tag"] / f"{smoke['output_tag']}.png",
    )
    print(json.dumps(serializable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
