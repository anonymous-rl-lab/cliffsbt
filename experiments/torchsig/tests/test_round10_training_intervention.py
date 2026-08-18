import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_round10_training_intervention",
    ROOT / "src" / "run_round10_training_intervention.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG = json.loads(
    (ROOT / "configs" / "round10_training_intervention_probe.json").read_text(encoding="utf-8")
)


class Round10TrainingInterventionTests(unittest.TestCase):
    def test_every_regime_preserves_count(self):
        rng = np.random.default_rng(31)
        for regime in CONFIG["round10"]["regime_order"]:
            theta, source = MODULE.sample_training_theta(regime, 1000, rng, CONFIG)
            self.assertEqual(theta.shape, (1000, 3))
            self.assertEqual(source.shape, (1000,))
            self.assertTrue(np.all(np.isfinite(theta)))

    def test_cliff_aware_support_is_more_targeted_than_random_broad(self):
        cliff, _ = MODULE.sample_training_theta(
            "cliff_aware", 10000, np.random.default_rng(41), CONFIG
        )
        broad, _ = MODULE.sample_training_theta(
            "random_broad", 10000, np.random.default_rng(42), CONFIG
        )
        self.assertLess(
            float(MODULE.nearest_path_distance(cliff, CONFIG).mean()),
            float(MODULE.nearest_path_distance(broad, CONFIG).mean()),
        )

    def test_shared_boundary_is_independent_of_model_tau(self):
        self.assertAlmostEqual(
            CONFIG["deployment"]["shared_risk_boundary"], 0.19984587722852512
        )

    def test_probe_result_integrity_when_present(self):
        path = ROOT / "results" / "round10_training_intervention_probe" / "checks.json"
        if not path.exists():
            return
        checks = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(checks["integrity_all_passed"])
        self.assertEqual(checks["metrics"]["expected_models"], 8)
        self.assertEqual(checks["metrics"]["fitted_models"], 8)

    def test_formal_protocol_uses_fresh_seeds_and_five_percent_dose(self):
        formal = json.loads(
            (ROOT / "configs" / "formal_round10_training_intervention_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(formal["round10"]["replicate_seeds"]), 5)
        self.assertTrue(
            set(formal["round10"]["replicate_seeds"]).isdisjoint(
                CONFIG["round10"]["replicate_seeds"]
            )
        )
        self.assertEqual(
            formal["round10"]["regimes"]["cliff_aware"]["enrichment_fraction"], 0.05
        )
        self.assertEqual(
            formal["round10"]["regimes"]["random_broad"]["enrichment_fraction"], 0.05
        )

    def test_formal_result_when_present(self):
        path = ROOT / "results" / "formal_round10_training_intervention_v1" / "checks.json"
        if not path.exists():
            return
        checks = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(checks["pretarget_all_passed"])
        self.assertTrue(checks["formal_all_passed"])
        self.assertEqual(checks["formal_metrics"]["baseline_crossing_fraction"], 1.0)
        self.assertEqual(checks["formal_metrics"]["cliff_aware_crossing_fraction"], 0.0)
        self.assertLess(checks["formal_metrics"]["cliff_aware_vs_random_end_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
