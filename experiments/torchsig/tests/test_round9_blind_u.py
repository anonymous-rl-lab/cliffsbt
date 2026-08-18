import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_round9_blind_u", ROOT / "src" / "run_round9_blind_u.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Round9BlindUTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.samples = self.rng.normal(size=(4, 16, 54))
        self.reference_samples = self.rng.normal(size=(80, 54))

    def test_relative_feature_shape_and_finiteness(self):
        indices = np.arange(25)
        reference = MODULE.reference_parameters(self.reference_samples, indices)
        features = MODULE.batch_features(
            self.samples, "train25_relative", reference
        )
        self.assertEqual(features.shape, (4, 52))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_wrong_reference_changes_relative_chart(self):
        indices = np.arange(25)
        reference = MODULE.reference_parameters(self.reference_samples, indices)
        shifted = MODULE.reference_parameters(self.reference_samples + 2.0, indices)
        first = MODULE.batch_features(self.samples, "train25_relative", reference)
        second = MODULE.batch_features(self.samples, "train25_relative", shifted)
        self.assertGreater(float(np.max(np.abs(first - second))), 0.1)

    def test_mean_channel_uses_no_physical_coordinate(self):
        indices = np.arange(25)
        reference = MODULE.reference_parameters(self.reference_samples, indices)
        observed = MODULE.batch_features(self.samples, "blind25_mean", reference)
        np.testing.assert_allclose(observed, self.samples[..., indices].mean(axis=-2))

    def test_moment_control_matches_declared_dimension(self):
        indices = np.arange(25)
        reference = MODULE.reference_parameters(self.reference_samples, indices)
        observed = MODULE.batch_features(self.samples, "blind25_moments", reference)
        self.assertEqual(observed.shape, (4, 50))
        self.assertTrue(np.all(np.isfinite(observed)))

    def test_formal_round9_result_when_present(self):
        path = ROOT / "results" / "formal_round9_blind_u_v1" / "checks.json"
        if not path.exists():
            return
        checks = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(checks["pretarget_all_passed"])
        self.assertFalse(checks["all_passed"])
        failed = {key for key, value in checks["gates"].items() if not value}
        self.assertEqual(
            failed,
            {
                "training_reference_warning_gain",
                "training_reference_warning_gain_interval",
            },
        )
        self.assertGreaterEqual(checks["metrics"]["train25_timely_warning_rate"], 0.35)
        self.assertEqual(checks["metrics"]["train25_false_alarm_rate"], 0.0)
        self.assertLessEqual(
            checks["metrics"]["training_reference_timely_gain_bootstrap_interval"][1],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
