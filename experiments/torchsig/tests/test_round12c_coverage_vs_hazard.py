import sys
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_round12c_coverage_vs_hazard_pilot import (  # noqa: E402
    cluster_bootstrap_ci,
    compute_seed_contrasts,
    local_band_hazard,
    select_coverage,
    select_global,
    selection_diagnostics,
)


def toy_candidates():
    rows = []
    sample = 0
    for label in range(4):
        for path in ["noise", "mixed_gradient"]:
            for phase in ["early", "late"]:
                for repeat in range(2):
                    rows.append(
                        {
                            "class_index": label,
                            "path": path,
                            "phase": phase,
                            "sample_index": sample,
                            "cell": f"class{label}|{path}|{phase}",
                            "random_score": float((sample * 17) % 31),
                            "local_hazard_score": float(100 - sample),
                            "local_incident": repeat == 0,
                        }
                    )
                    sample += 1
    return pd.DataFrame(rows)


class Round12CCoverageVersusHazardTests(unittest.TestCase):
    def test_coverage_selector_fills_all_sixteen_cells_with_unique_sources(self):
        selected = select_coverage(
            toy_candidates(),
            4,
            ["noise", "mixed_gradient"],
            ["early", "late"],
            "random_score",
        )
        diagnostics = selection_diagnostics(selected, 16)
        self.assertEqual(len(selected), 16)
        self.assertEqual(diagnostics["coverage_fraction"], 1.0)
        self.assertEqual(diagnostics["unique_source_trajectories"], 16)

    def test_global_hazard_selector_uses_largest_unique_scores(self):
        candidates = toy_candidates()
        selected = select_global(candidates, 16, "local_hazard_score")
        self.assertEqual(len(selected), 16)
        self.assertGreaterEqual(selected["local_hazard_score"].min(), 85.0)

    def test_local_hazard_detects_persistent_crossing(self):
        margin = np.asarray([0.5, 0.4, 0.2, -0.1, -0.2, -0.3])
        score, chosen, incident, persistence, descent, negative_area = local_band_hazard(
            margin, 2, 5, 2
        )
        self.assertTrue(incident)
        self.assertEqual(chosen, 3)
        self.assertEqual(persistence, 1.0)
        self.assertGreater(score, 6.0)
        self.assertGreater(descent, 0)
        self.assertGreater(negative_area, 0)

    def test_seed_contrasts_have_expected_sign(self):
        rows = []
        for seed in [1, 2, 3]:
            for arm, value in {
                "random_unstratified": 0.1,
                "hazard_concentrated": 0.2,
                "coverage_random": 0.3,
                "coverage_hazard": 0.35,
            }.items():
                rows.append(
                    {
                        "replicate_seed": seed,
                        "arm": arm,
                        "mean_end_risk_reduction": value,
                    }
                )
        contrasts = compute_seed_contrasts(pd.DataFrame(rows))
        self.assertTrue(
            np.allclose(contrasts["coverage_random_minus_hazard_concentrated"], 0.1)
        )
        self.assertTrue(np.allclose(contrasts["coverage_gain_at_high_hazard"], 0.15))
        self.assertTrue(np.allclose(contrasts["hazard_gain_at_high_coverage"], 0.05))

    def test_cluster_bootstrap_is_seed_clustered(self):
        lower, upper = cluster_bootstrap_ci(np.asarray([0.1, 0.2, 0.3]), 2000, 7)
        self.assertTrue(0.09 <= lower <= 0.2)
        self.assertTrue(0.2 <= upper <= 0.31)

    def test_fresh_pilot_result_when_present(self):
        path = ROOT / "results" / "round12c_coverage_vs_hazard_pilot" / "summary.json"
        if not path.exists():
            self.skipTest("Round 12C result is not packaged")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["decision"], "PILOT_SUPPORTS_COVERAGE_DOMINANCE")
        self.assertEqual(summary["checks"]["passed"], summary["checks"]["total"])
        values = summary["checks"]["values"]
        self.assertGreater(values["hazard_score_advantage"], 0)
        self.assertEqual(values["coverage_random_mean_coverage_fraction"], 1.0)
        self.assertGreater(values["primary_seed_cluster_ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
