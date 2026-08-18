import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_precursor_order_knockout",
    ROOT / "src" / "probe_precursor_order_knockout.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrecursorMechanismTests(unittest.TestCase):
    def test_fixed_terminal_permutation_removes_order_signal(self):
        times = np.arange(10)
        ordered = np.linspace(0.10, 0.28, 10)
        shuffled = ordered.copy()
        shuffled[:-1] = shuffled[:-1][::-1]
        np.testing.assert_array_equal(np.sort(shuffled), np.sort(ordered))
        self.assertEqual(shuffled[-1], ordered[-1])
        ordered_slopes, ordered_scores = MODULE.forecast_window(
            ordered, times, crossing=10, history=6, horizon=5
        )
        shuffled_slopes, shuffled_scores = MODULE.forecast_window(
            shuffled, times, crossing=10, history=6, horizon=5
        )
        self.assertGreater(ordered_slopes.mean(), shuffled_slopes.mean())
        self.assertGreater(ordered_scores.mean(), shuffled_scores.mean())

    def test_formal_mechanism_result_when_present(self):
        path = (
            ROOT
            / "results"
            / "formal_precursor_mechanism_knockout_v3"
            / "checks.json"
        )
        if not path.exists():
            return
        checks = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(checks["pretarget_all_passed"])
        self.assertTrue(checks["all_passed"])
        self.assertEqual(checks["metrics"]["multiset_invariant_violations"], 0)
        self.assertEqual(
            checks["metrics"]["terminal_state_invariant_violations"], 0
        )
        self.assertGreater(
            checks["metrics"]["slope_difference_bootstrap_interval"][0], 0
        )
        self.assertGreater(
            checks["metrics"]["forecast_difference_bootstrap_interval"][0], 0
        )


if __name__ == "__main__":
    unittest.main()
