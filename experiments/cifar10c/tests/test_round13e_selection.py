#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parents[1] / "src" / "run_round13e_repair.py"
SPEC = importlib.util.spec_from_file_location("round13e", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_round_robin_spans_more_fragments() -> None:
    candidates = []
    for family_index, family in enumerate(MODULE.CORRUPTIONS[:4]):
        for label in range(3):
            for identity in range(10):
                candidates.append({"identity": family_index * 1000 + label * 100 + identity, "corruption": family, "corruption_index": family_index, "severity": label + 1, "label": label, "hazard_score": 100.0 - family_index * 20 - label * 3 - identity})
    labels = np.arange(5000) % 10
    calibration = np.arange(100)
    selected = MODULE.select_sets(candidates, calibration, labels, 24, 1)
    candidate_keys = {(r["identity"], r["corruption"], r["severity"]) for r in candidates}
    hazard = MODULE.selection_stats(selected["hazard"], candidate_keys)
    coverage = MODULE.selection_stats(selected["coverage"], candidate_keys)
    assert hazard["hazard_hit_rate"] == coverage["hazard_hit_rate"] == 1.0
    assert coverage["unique_fragments"] > hazard["unique_fragments"]


if __name__ == "__main__":
    test_round_robin_spans_more_fragments()
    print("1/1 tests passed")

