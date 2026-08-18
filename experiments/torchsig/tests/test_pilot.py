import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_pilot", ROOT / "src" / "run_pilot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DESIGN_SPEC = importlib.util.spec_from_file_location(
    "measurement_design", ROOT / "src" / "measurement_design.py"
)
DESIGN = importlib.util.module_from_spec(DESIGN_SPEC)
DESIGN_SPEC.loader.exec_module(DESIGN)


class TorchSigPilotTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "configs" / "pilot.json").read_text())

    def test_mechanism_severity_is_monotone(self):
        mild = MODULE.mechanism_values(np.array([-0.75, -0.75, -0.75]))
        severe = MODULE.mechanism_values(np.array([0.75, 0.75, 0.75]))
        self.assertGreater(mild["snr_db"], severe["snr_db"])
        self.assertLess(mild["phase_noise_degrees"], severe["phase_noise_degrees"])
        self.assertGreater(mild["amplifier_psat_backoff"], severe["amplifier_psat_backoff"])

    def test_deterministic_signal_and_feature(self):
        theta = np.zeros(3)
        x1 = MODULE.generate_iq("qpsk", theta, np.random.default_rng(123), self.cfg["signal"])
        x2 = MODULE.generate_iq("qpsk", theta, np.random.default_rng(123), self.cfg["signal"])
        np.testing.assert_array_equal(x1, x2)
        f = MODULE.extract_features(x1)
        self.assertEqual(f.shape, (len(MODULE.FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(f)))

    def test_quadratic_parameterization(self):
        tau = 0.3
        b = np.array([0.7, -0.2, 0.1])
        h = np.array([[1.2, 0.3, -0.1], [0.3, -0.4, 0.2], [-0.1, 0.2, 0.8]])
        u = np.array([0.05, -0.03, 0.02])
        coefficients = np.array(
            [b[0], b[1], b[2], h[0, 0], h[1, 1], h[2, 2], h[0, 1], h[0, 2], h[1, 2]]
        )
        via_design = tau + MODULE.quadratic_design(u)[0] @ coefficients
        direct = MODULE.quadratic_risk(u, tau, b, h)
        self.assertAlmostEqual(via_design, direct, places=12)

    def test_quadratic_pair_optimizer(self):
        cfg = {
            "master_seed": 17,
            "target": {"risk_margin_gamma": 0.04},
            "risk_surface": {
                "mechanism_ball_radius": 0.12,
                "remainder_buffer": 0.01,
            },
        }
        q = np.diag([2.0, 1.0, 0.5])
        q_pinv = np.linalg.inv(q)
        b = np.array([1.0, 0.4, 0.1])
        h = np.diag([0.5, 0.2, -0.1])
        pair = MODULE.optimize_supporting_pair(0.25, b, h, q, q_pinv, cfg)
        self.assertLessEqual(pair["optimizer_constraint_error"], 1e-6)
        self.assertGreaterEqual(pair["support_slack_min"], -1e-3)
        self.assertLessEqual(np.linalg.norm(pair["u_minus"]), 0.120001)
        self.assertLessEqual(np.linalg.norm(pair["u_plus"]), 0.120001)

    def test_wilson_interval_contains_point_estimate(self):
        low, high = MODULE.wilson_interval(30, 100)
        self.assertLess(low, 0.30)
        self.assertGreater(high, 0.30)

    def test_result_gate_file_when_present(self):
        for path in (ROOT / "results").glob("*/checks.json"):
            checks = json.loads(path.read_text())
            self.assertIn("all_passed", checks)
            if "gates" in checks and "risk_range" in checks["gates"]:
                self.assertIn("risk_range", checks["gates"])
                self.assertTrue(
                    "risk_linearity" in checks["gates"]
                    or "risk_surface_fit" in checks["gates"]
                )
            elif "gates" in checks:
                self.assertTrue(checks["gates"])
                self.assertTrue(
                    "pretarget_all_passed" in checks or "pretarget" in checks
                )
            else:
                self.assertIn("pretarget_all_passed", checks)
                self.assertIn("curve_gates", checks)
                self.assertIn("mean_accuracy_loss_gates", checks)

    def test_measurement_compression_result_when_present(self):
        base = ROOT / "results" / "formal_measurement_compression_v1"
        if base.exists():
            pretarget = json.loads(
                (base / "pretarget_measurement_compression_gate.json").read_text()
            )
            checks = json.loads((base / "checks.json").read_text())
            self.assertTrue(pretarget["all_passed"])
            self.assertTrue(checks["all_passed"])
            for center in pretarget["centers"].values():
                self.assertLessEqual(center["metrics"]["cost_fraction"], 0.50)
                self.assertGreaterEqual(
                    center["metrics"]["information_retention"], 0.80
                )

    def test_quadratic_ledger_uses_relative_risk_semantics(self):
        path = ROOT / "results" / "formal_quadratic_v3" / "quadratic_v3_ledger.json"
        if path.exists():
            ledger = json.loads(path.read_text())
            self.assertIn("relative", ledger["semantics"])
            self.assertIn("lower_risk_state", ledger["revealed"])
            self.assertIn("higher_risk_state", ledger["revealed"])
            self.assertNotIn("risk_minus", ledger["revealed"])

    def test_measurement_groups_partition_rich_channel(self):
        DESIGN.validate_groups(54)
        self.assertEqual(len(DESIGN.all_group_subsets()), 63)
        self.assertEqual(len(DESIGN.group_indices(tuple(DESIGN.MEASUREMENT_GROUPS))), 54)

    def test_predicted_batch_size_decreases_with_information(self):
        weak = DESIGN.predicted_batch_size(0.2, 0.90)
        strong = DESIGN.predicted_batch_size(1.0, 0.90)
        self.assertIsNotNone(weak)
        self.assertIsNotNone(strong)
        self.assertGreater(weak, strong)


if __name__ == "__main__":
    unittest.main()
