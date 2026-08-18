#!/usr/bin/env python3
"""End-to-end Phase-2 test using explicitly synthetic predictions only."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from common import MODEL_SEEDS, PACKAGE, SCHEDULE_IDS, TARGET_FAMILIES, sha256, write_json
from run_phase2 import confirmation_truth, phase2


def main() -> None:
    if not (PACKAGE / "PACKAGE_MANIFEST.sha256").exists():
        raise SystemExit("freeze the package manifest before the synthetic Phase-2 test")
    truth = confirmation_truth()
    shape = (5, 3, 10, 13, 50)
    baseline = np.empty(shape, dtype=np.int8)
    repaired = np.empty(shape, dtype=np.int8)
    alarms = []
    for si, seed in enumerate(MODEL_SEEDS):
        for qi, schedule in enumerate(SCHEDULE_IDS):
            for fi, family in enumerate(TARGET_FAMILIES):
                target = truth[family]
                base = np.tile(target, (13, 1))
                repair = base.copy()
                if fi < 6:
                    base[6:, :30] = (base[6:, :30] + 1) % 10
                    repair[6:, :20] = (repair[6:, :20] + 1) % 10
                    alarm = 4
                else:
                    alarm = None
                baseline[si, qi, fi] = base
                repaired[si, qi, fi] = repair
                alarms.append({
                    "seed": seed,
                    "schedule_id": schedule,
                    "family": family,
                    "hybrid25_scores": [0.0] * 13,
                    "hybrid25_alarm": alarm,
                })
    with tempfile.TemporaryDirectory(prefix="cure_or_v2_synthetic_") as raw:
        workspace = Path(raw)
        metadata = workspace / "run_metadata.json"
        assessment = workspace / "calibration_assessment.json"
        alarm_file = workspace / "blind_alarm_rows.json"
        predictions = workspace / "blind_predictions.npz"
        write_json(metadata, {
            "package_manifest_sha256": sha256(PACKAGE / "PACKAGE_MANIFEST.sha256"),
            "confirmation_labels_scored": False,
        })
        write_json(assessment, {
            "warning": {"synthetic": True},
            "repair_by_model": {str(seed): {"eligible": True} for seed in MODEL_SEEDS},
        })
        write_json(alarm_file, alarms)
        np.savez_compressed(
            predictions,
            baseline_predicted=baseline,
            candidate_repair_predicted=repaired,
            deployed_repair_predicted=repaired,
            hybrid25_sensors=np.zeros((5, 3, 10, 13, 25)),
            warning_scores=np.zeros((5, 3, 10, 13)),
            seeds=np.asarray(MODEL_SEEDS),
            schedule_ids=np.asarray(SCHEDULE_IDS),
            families=np.asarray(TARGET_FAMILIES),
        )
        committed = [metadata, assessment, alarm_file, predictions]
        (workspace / "BLIND_OUTPUTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in committed), encoding="utf-8",
        )
        result = phase2(workspace)
        if result["overall_decision"] != "FORMATION_WARNING_REPAIR_CONFIRMED":
            raise AssertionError(result["overall_decision"])
        if not all(result[key]["pass"] for key in ("h1_formation", "h2_warning", "h3_repair")):
            raise AssertionError("synthetic end-to-end gates did not all pass")
    print("SYNTHETIC PHASE2 END-TO-END PASS")


if __name__ == "__main__":
    main()
