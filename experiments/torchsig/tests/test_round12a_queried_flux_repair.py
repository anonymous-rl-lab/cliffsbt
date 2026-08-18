import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from torchsig_official_source_numpy_runtime import install

if "torchsig" not in sys.modules:
    install()

from run_round12a_queried_flux_repair_v3 import true_class_balanced_prefix


class Round12AQueriedFluxRepairTests(unittest.TestCase):
    def test_equal_true_class_training_quota_after_query(self):
        frame = pd.DataFrame(
            {
                "class_index": [0] * 5 + [1] * 5,
                "queried_true_flux_score": [5, 4, 3, 2, 1] * 2,
                "path": ["p"] * 10,
                "sample_index": list(range(10)),
            }
        )
        selected = true_class_balanced_prefix(
            frame, budget=4, classes=2, score_column="queried_true_flux_score"
        )
        self.assertEqual(selected["class_index"].value_counts().to_dict(), {0: 2, 1: 2})


if __name__ == "__main__":
    unittest.main()

