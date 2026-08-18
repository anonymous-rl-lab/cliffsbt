import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from torchsig_official_source_numpy_runtime import install

if "torchsig" not in sys.modules:
    install()

from run_round12a_blind_flux_repair import rank_unlabeled, replace_same_class


class Round12ABlindFluxRepairTests(unittest.TestCase):
    def test_blind_ranker_has_no_label_or_control_argument(self):
        names = set(inspect.signature(rank_unlabeled).parameters)
        self.assertNotIn("labels", names)
        self.assertNotIn("theta", names)
        self.assertNotIn("u", names)

    def test_persistent_switch_ranks_above_stable_trace(self):
        probability = np.asarray(
            [
                [[0.90, 0.10], [0.80, 0.20]],
                [[0.80, 0.20], [0.79, 0.21]],
                [[0.40, 0.60], [0.78, 0.22]],
                [[0.30, 0.70], [0.77, 0.23]],
            ]
        )
        ranked = rank_unlabeled({"path": probability}, 0, 2, "blind_flux", 1)
        self.assertEqual(int(ranked.iloc[0]["sample_index"]), 0)

    def test_same_class_replacement_preserves_counts(self):
        features = np.arange(24, dtype=float).reshape(8, 3)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        repair_features = np.asarray([[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])
        repair_labels = np.asarray([0, 1])
        _, repaired_labels, indices = replace_same_class(
            features, labels, repair_features, repair_labels, 7
        )
        self.assertEqual(len(indices), 2)
        np.testing.assert_array_equal(np.bincount(repaired_labels), np.bincount(labels))


if __name__ == "__main__":
    unittest.main()
