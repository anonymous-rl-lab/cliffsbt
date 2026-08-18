import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class Round11BoundaryFluxTests(unittest.TestCase):
    def test_round11a_registered_stop_is_retained(self):
        summary = json.loads(
            (ROOT / "results/round11_paired_sync_smoke/summary.json").read_text()
        )
        self.assertEqual(summary["decision"], "SMOKE_STOP_REDESIGN")
        self.assertEqual(summary["checks"]["passed"], 6)
        self.assertEqual(summary["checks"]["total"], 7)
        self.assertFalse(summary["checks"]["checks"]["velocity_matching_specificity"])

    def test_round11b_is_compatibility_probe_not_formal(self):
        summary = json.loads(
            (ROOT / "results/round11b_distributed_flux_probe/summary.json").read_text()
        )
        self.assertEqual(
            summary["decision"],
            "DISTRIBUTED_FLUX_COMPAT_PROBE_PASS_STANDARD_RUNTIME_REQUIRED",
        )
        self.assertFalse(summary["standard_torchsig_runtime"])
        self.assertEqual(summary["checks"]["passed"], 9)
        self.assertEqual(summary["checks"]["total"], 9)

    def test_paired_flux_identity(self):
        for tag in (
            "round11_paired_sync_smoke",
            "round11b_distributed_flux_probe",
            "round11c_official_source_flux",
        ):
            frame = pd.read_csv(ROOT / f"results/{tag}/transition_flux.csv")
            self.assertLessEqual(float(frame["accounting_error"].max()), 1e-12)

    def test_fresh_redesign_seed_and_boundary_controls(self):
        first = json.loads((ROOT / "configs/round11_paired_sync_smoke.json").read_text())
        second = json.loads((ROOT / "configs/round11b_distributed_flux_probe.json").read_text())
        self.assertTrue(set(first["replicate_seeds"]).isdisjoint(second["replicate_seeds"]))
        frame = pd.read_csv(ROOT / "results/round11b_distributed_flux_probe/path_summary.csv")
        baseline = frame[frame["regime"] == "baseline"]
        self.assertTrue(bool(baseline["above_random_boundary_q95"].all()))
        self.assertFalse(bool(baseline["above_velocity_shuffle_q95"].any()))

    def test_round11c_source_faithful_confirmation_and_intervals(self):
        summary = json.loads(
            (ROOT / "results/round11c_official_source_flux/summary.json").read_text()
        )
        self.assertEqual(
            summary["decision"],
            "OFFICIAL_SOURCE_NUMPY_CONFIRMATION_PASS_PACKAGE_RUNTIME_REQUIRED",
        )
        self.assertFalse(summary["standard_torchsig_package_runtime"])
        self.assertEqual(summary["checks"]["passed"], 12)
        self.assertEqual(summary["checks"]["total"], 12)
        values = summary["checks"]["values"]
        self.assertGreater(values["end_risk_reduction_cluster_ci95"][0], 0)
        self.assertGreater(values["incident_crossing_reduction_cluster_ci95"][0], 0)
        self.assertEqual(
            summary["runtime"]["torchsig_source_commit"],
            "d9abfe1af2b0216d2bacc31c677407ed31878086",
        )


if __name__ == "__main__":
    unittest.main()
