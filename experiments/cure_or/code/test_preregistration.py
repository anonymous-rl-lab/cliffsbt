#!/usr/bin/env python3
"""Outcome-free unit and design tests for the v2 experiment package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import (
    CONFIG, MODEL_SEEDS, PACKAGE, SCHEDULE_IDS, TARGET_FAMILIES, anchored_update,
    assigned_levels, exact_flow, hybrid25, persistent_cliff, read_json,
    score_hybrid,
)
from run_phase1 import fixed_repair_rows, repair_calibration_guard, typed_rows


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def synthetic_item(schedule_id: int, family: int, risk: list[float]) -> dict:
    return {"schedule_id": schedule_id, "family": family, "risk": risk}


def main() -> None:
    check("registered model seeds are unique", len(set(MODEL_SEEDS)) == 5)
    check("three schedule variants are frozen", len(set(SCHEDULE_IDS)) == 3)
    bases = [(index % 10, index // 10, index % 3) for index in range(50)]
    for schedule_id in SCHEDULE_IDS:
        check(
            f"schedule {schedule_id} ends at severity four",
            set(assigned_levels(bases, TARGET_FAMILIES[0], 12, schedule_id).values()) == {4},
        )
    check(
        "schedule variants change asynchronous identity order",
        assigned_levels(bases, TARGET_FAMILIES[0], 1, SCHEDULE_IDS[0])
        != assigned_levels(bases, TARGET_FAMILIES[0], 1, SCHEDULE_IDS[1]),
    )

    status = np.asarray([[False, False], [True, False], [True, True]], dtype=bool)
    forward, recovery, residual, closure = exact_flow(status)
    check("paired boundary-flow identity closes", closure <= 1e-12 and np.allclose(residual, 0))
    check("persistent cliff definition", persistent_cliff([0.2, 0.6, 0.4, 0.6, 0.7]) == 3)
    check("transient crossing is not a cliff", persistent_cliff([0.2, 0.6, 0.4, 0.3]) is None)

    flux = np.arange(25, dtype=float) / 25.0
    moments = np.arange(25, dtype=float) / 10.0
    sensor = hybrid25(flux, moments)
    check("Hybrid25 has exactly 25 channels", sensor.shape == (25,))
    model = read_json(CONFIG / "hybrid25_warning_model_frozen.json")
    scores = score_hybrid(np.tile(sensor, (13, 1)), model)
    check("frozen Hybrid25 readout emits 13 finite scores", scores.shape == (13,) and np.isfinite(scores).all())

    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 8))
    labels = np.arange(20) % 10
    base = {"mean": np.zeros(8), "scale": np.ones(8), "weights": np.zeros((9, 10))}
    repaired = anchored_update(base, features, labels, 1000.0)
    check("anchored repair preserves head shape", repaired["weights"].shape == base["weights"].shape)

    candidates = typed_rows(PACKAGE / "data" / "REPAIR_CANDIDATES_FROZEN.csv")
    selected = fixed_repair_rows(candidates, 5)
    check("repair covers 100 fragments", len({row["fragment"] for row in selected}) == 100)
    check("repair budget is exactly 500", len(selected) == 500)

    safe, improved = [], []
    for schedule_id in SCHEDULE_IDS:
        for family in TARGET_FAMILIES:
            base_risk = [0.2] * 6 + ([0.6] * 7 if family == TARGET_FAMILIES[0] else [0.3] * 7)
            repaired_risk = [max(0.0, value - 0.15) for value in base_risk]
            safe.append(synthetic_item(schedule_id, family, base_risk))
            improved.append(synthetic_item(schedule_id, family, repaired_risk))
    guard_spec = read_json(CONFIG / "repair_protocol.json")["calibration_guard_per_model"]
    decision = repair_calibration_guard(safe, improved, guard_spec)
    check("repair guard accepts safe cliff removal", decision["eligible"])
    worsened = [dict(item) for item in improved]
    worsened[1] = synthetic_item(SCHEDULE_IDS[0], TARGET_FAMILIES[1], [0.4] * 13)
    check("repair guard rejects a path-risk violation", not repair_calibration_guard(safe, worsened, guard_spec)["eligible"])

    freeze = subprocess.run(
        [sys.executable, str(PACKAGE / "code" / "freeze_design.py"), "--verify"],
        check=False,
    )
    check("frozen design verifier", freeze.returncode == 0)
    phase1_source = (PACKAGE / "code" / "run_phase1.py").read_text(encoding="utf-8")
    check("phase 1 does not call confirmation_truth", "confirmation_truth" not in phase1_source)
    print("ALL OUTCOME-FREE TESTS PASSED")


if __name__ == "__main__":
    main()
