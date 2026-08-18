# Round 13D pretarget schema correction

The v1 protocol and evaluator expected `labels.npy` to have shape `(10000,)`.
After the official archive passed its MD5 check and was extracted, a file-shape
audit showed that the archive stores shape `(50000,)`: the 10,000 CIFAR-10 test
labels are repeated once for each of the five severity blocks.

No checkpoint had been loaded and no model prediction, risk curve, crossing,
endpoint, or gate had been computed when this discrepancy was found. The v1
files were copied verbatim to `pretarget_snapshots/round13d_v1/`, where their
hashes match `ROUND13D_PRETARGET_SHA256.txt`.

The only v2 change is to require 50,000 labels and verify that its five 10,000-
entry blocks are exactly identical before using the first block. All scientific
endpoints, corruption families, seeds, thresholds, controls, and gates remain
unchanged.

