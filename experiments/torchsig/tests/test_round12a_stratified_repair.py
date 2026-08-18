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

from run_round12a_blind_flux_repair_v2 import stratified_prefix


class Round12AStratifiedRepairTests(unittest.TestCase):
    def test_equal_anchor_prediction_quota(self):
        frame = pd.DataFrame(
            {
                "anchor_predicted_class": [0] * 5 + [1] * 5,
                "score": [5, 4, 3, 2, 1] * 2,
                "path": ["p"] * 10,
                "sample_index": list(range(10)),
            }
        )
        selected = stratified_prefix(frame, budget=4, classes=2)
        self.assertEqual(selected["anchor_predicted_class"].value_counts().to_dict(), {0: 2, 1: 2})


if __name__ == "__main__":
    unittest.main()

