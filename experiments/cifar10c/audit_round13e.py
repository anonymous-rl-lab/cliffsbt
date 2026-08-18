#!/usr/bin/env python3
"""Independent evidence audit for the frozen Round 13E repair experiment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "results/round13e_formal_v1"
SMOKE = ROOT / "results/round13e_smoke_v1"
SEEDS = (71, 83, 97)
ARMS = ("baseline", "random", "hazard", "coverage")
CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_pretarget() -> bool:
    pattern = re.compile(r"^([0-9a-f]{64})\s+(.+)$")
    found = 0
    for line in (ROOT / "ROUND13E_PRETARGET_SHA256.txt").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        found += 1
        if digest(ROOT / match.group(2)) != match.group(1):
            return False
    return found == 3


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(20000, len(values)))
    dist = values[sampled].mean(axis=1)
    return float(values.mean()), [float(np.quantile(dist, 0.025)), float(np.quantile(dist, 0.975))]


def close(left: float, right: float, tol: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tol


def main() -> None:
    formal = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    smoke = json.loads((SMOKE / "summary.json").read_text(encoding="utf-8"))
    selections_doc = json.loads((FORMAL / "selections.json").read_text(encoding="utf-8"))
    selections = selections_doc["selections"]
    calibration = np.asarray(selections_doc["calibration"], dtype=int)
    holdout = np.asarray(selections_doc["holdout"], dtype=int)

    reference = np.load(ROOT / "results/cifar10c_official_v1/paired_outputs.npz")
    ref_pred = reference["predictions"]
    ref_margin = reference["margins"]
    labels = reference["labels"]
    wrong_votes = (ref_pred != labels[None, None, None, :]).sum(axis=0)
    majority_wrong = wrong_votes >= 2
    mean_margin = ref_margin.mean(axis=0)
    candidate = {}
    for c_index, corruption in enumerate(CORRUPTIONS):
        for identity in calibration:
            if majority_wrong[c_index, 0, identity]:
                continue
            for severity in range(1, 6):
                if not majority_wrong[c_index, severity - 1, identity] and majority_wrong[c_index, severity, identity]:
                    candidate[(int(identity), corruption, severity)] = float(-mean_margin[c_index, severity, identity])
                    break

    selection_hit = {}
    for arm in ("random", "hazard", "coverage"):
        rows = selections[arm]
        selection_hit[arm] = np.mean([(row["identity"], row["corruption"], row["severity"]) in candidate for row in rows])
    expected_hazard_keys = {
        key for key, _ in sorted(candidate.items(), key=lambda item: (-item[1], CORRUPTIONS.index(item[0][1]), item[0][0]))[:1000]
    }
    observed_hazard_keys = {(row["identity"], row["corruption"], row["severity"]) for row in selections["hazard"]}

    raw_shape_ok = True
    metric_error = 0.0
    accounting_error = 0.0
    checkpoint_ok = True
    observed_metrics: dict[str, dict[str, dict[str, float]]] = {str(seed): {} for seed in SEEDS}
    for seed in SEEDS:
        for arm in ARMS:
            fit = FORMAL / "fits" / f"seed{seed}_{arm}"
            raw = np.load(fit / "paired_outputs.npz")
            pred = raw["predictions"]
            y = raw["labels"]
            raw_shape_ok &= pred.shape == (15, 6, 8000)
            raw_shape_ok &= np.array_equal(raw["holdout"], holdout)
            raw_shape_ok &= np.array_equal(raw["corruptions"].astype(str), np.asarray(CORRUPTIONS))
            raw_shape_ok &= np.array_equal(y, labels[holdout])
            wrong = pred != y[None, None, :]
            errors = wrong.mean(axis=2)
            incident = np.logical_and(~wrong[:, :-1], wrong[:, 1:]).mean(axis=2)
            recovery = np.logical_and(wrong[:, :-1], ~wrong[:, 1:]).mean(axis=2)
            accounting_error = max(accounting_error, float(np.max(np.abs(np.diff(errors, axis=1) - (incident - recovery)))))
            beta = float(formal["results"][str(seed)][arm]["beta"])
            recomputed = {
                "clean_error": float(errors[:, 0].mean()),
                "endpoint_error_mean": float(errors[:, -1].mean()),
                "risk_area_mean": float(errors[:, 1:].mean()),
                "crossing_fraction": float(np.mean(np.any(errors >= beta, axis=1))),
                "endpoint_net_flux_mean": float((errors[:, -1] - errors[:, 0]).mean()),
            }
            observed_metrics[str(seed)][arm] = recomputed
            stored = formal["results"][str(seed)][arm]["metrics"]
            metric_error = max(metric_error, *(abs(recomputed[key] - stored[key]) for key in recomputed))
            checkpoint = torch.load(fit / "model.pt", map_location="cpu", weights_only=False)
            checkpoint_ok &= checkpoint["seed"] == seed and checkpoint["arm"] == arm
            checkpoint_ok &= len(checkpoint["train_indices"]) == 20000
            checkpoint_ok &= len(checkpoint["state_dict"]) > 0

    comparison_ok = True
    metrics = ("endpoint_error_mean", "risk_area_mean", "crossing_fraction", "clean_error")
    others = ("hazard", "baseline", "random")
    comparison_index = 0
    recomputed_comparisons = {}
    for metric_name in metrics:
        coverage = np.asarray([observed_metrics[str(seed)]["coverage"][metric_name] for seed in SEEDS])
        for other in others:
            other_values = np.asarray([observed_metrics[str(seed)][other][metric_name] for seed in SEEDS])
            differences = coverage - other_values
            estimate, ci95 = bootstrap(differences, 1400 + comparison_index)
            key = f"coverage_minus_{other}_{metric_name}"
            recomputed_comparisons[key] = {"estimate": estimate, "ci95": ci95}
            stored = formal["comparisons"][key]
            comparison_ok &= close(estimate, stored["estimate"])
            comparison_ok &= np.allclose(ci95, stored["ci95"], atol=1e-12, rtol=0)
            comparison_ok &= np.allclose(differences, stored["seed_differences"], atol=1e-12, rtol=0)
            comparison_index += 1

    stats = formal["selection_stats"]
    all_clean_competent = all(1 - observed_metrics[str(seed)][arm]["clean_error"] >= 0.45 for seed in SEEDS for arm in ARMS)
    gates = {
        "equal_dangerous_hit_rate": selection_hit["hazard"] == selection_hit["coverage"] == 1.0 and len(selections["hazard"]) == len(selections["coverage"]) == 1000,
        "coverage_separation": stats["coverage"]["unique_fragments"] >= 2 * stats["hazard"]["unique_fragments"] and stats["coverage"]["families"] >= 14,
        "model_competence": all_clean_competent,
        "exact_paired_accounting": accounting_error <= 1e-12,
        "coverage_beats_hazard_endpoint": recomputed_comparisons["coverage_minus_hazard_endpoint_error_mean"]["ci95"][1] < 0,
        "coverage_beats_hazard_area": recomputed_comparisons["coverage_minus_hazard_risk_area_mean"]["ci95"][1] < 0,
        "coverage_reduces_crossing": recomputed_comparisons["coverage_minus_hazard_crossing_fraction"]["estimate"] < 0,
        "coverage_beats_baseline_endpoint": recomputed_comparisons["coverage_minus_baseline_endpoint_error_mean"]["ci95"][1] < 0,
        "clean_risk_guard": recomputed_comparisons["coverage_minus_hazard_clean_error"]["ci95"][1] <= 0.02,
    }

    checks = {
        "pretarget_protocol_and_code_hashes": verify_pretarget(),
        "smoke_7_of_7_advance": smoke["decision"] == "ADVANCE_TO_FORMAL" and smoke["gate_count"] == smoke["gate_total"] == 7,
        "formal_state_has_all_12_fits": len(json.loads((FORMAL / "state.json").read_text())["completed"]) == 12,
        "calibration_holdout_disjoint_and_complete": len(calibration) == 2000 and len(holdout) == 8000 and len(np.intersect1d(calibration, holdout)) == 0 and len(np.union1d(calibration, holdout)) == 10000,
        "equal_declared_budgets": all(len(selections[arm]) == 1000 for arm in ("random", "hazard", "coverage")),
        "independent_dangerous_hit_rates": close(selection_hit["random"], 0.075) and selection_hit["hazard"] == selection_hit["coverage"] == 1.0,
        "hazard_is_exact_top_1000": observed_hazard_keys == expected_hazard_keys,
        "coverage_fragment_manipulation": stats["hazard"]["unique_fragments"] == 262 and stats["coverage"]["unique_fragments"] == 731,
        "raw_outputs_identity_and_shape": bool(raw_shape_ok),
        "independent_metric_reconstruction": metric_error <= 1e-12,
        "independent_signed_flux_accounting": accounting_error <= 1e-12,
        "all_12_checkpoints_complete": bool(checkpoint_ok),
        "paired_bootstrap_reconstruction": bool(comparison_ok),
        "formal_gates_reconstructed": gates == formal["gates"],
        "formal_decision_6_of_9": formal["decision"] == "PARTIAL_OR_STOP" and formal["gate_count"] == 6 and formal["gate_total"] == 9,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "max_metric_reconstruction_error": metric_error,
        "max_independent_accounting_error": accounting_error,
        "independent_hazard_hit_rates": {key: float(value) for key, value in selection_hit.items()},
        "reconstructed_gates": gates,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
