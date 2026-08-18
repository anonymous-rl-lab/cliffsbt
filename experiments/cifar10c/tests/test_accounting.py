#!/usr/bin/env python3
"""Small deterministic checks for the paired accounting implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "src" / "run_cifar10_paired_smoke.py"
SPEC = importlib.util.spec_from_file_location("round13_smoke", MODULE)
assert SPEC and SPEC.loader
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_forward_minus_recovery_equals_error_change() -> None:
    y = np.array([0, 0, 0, 0])
    preds = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
        ]
    )
    margins = np.where(preds == y[None, :], 1.0, -1.0)
    result = SMOKE.analyze_path(preds, margins, y)
    assert result["max_abs_accounting_error"] == 0.0
    assert abs(result["cumulative_net_flux"][-1] - (result["errors"][-1] - result["errors"][0])) < 1e-15


def test_first_crossings_are_distributed() -> None:
    y = np.zeros(5, dtype=int)
    preds = np.zeros((6, 5), dtype=int)
    for sample, level in enumerate([1, 2, 3, 4, 5]):
        preds[level:, sample] = 1
    margins = np.where(preds == 0, 1.0, -1.0)
    result = SMOKE.analyze_path(preds, margins, y)
    assert result["first_crossing_counts"] == [1, 1, 1, 1, 1]
    assert abs(result["first_crossing_entropy"] - 1.0) < 1e-12
    assert result["endpoint_persistence"] == 1.0


if __name__ == "__main__":
    test_forward_minus_recovery_equals_error_change()
    test_first_crossings_are_distributed()
    print("2/2 tests passed")

