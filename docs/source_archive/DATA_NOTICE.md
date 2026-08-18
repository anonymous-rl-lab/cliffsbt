# Data notice

## Covertype

The package includes the public UCI Covertype source data and the exact
scikit-learn cache used by the later runs. Attribution, license information,
source URLs, source hashes, class filtering, window construction, and panel
semantics are in `covertype/DATA_NOTICE.md`. The unified audit checks the three
recorded source hashes.

Covertype elevation windows are balanced cross-sectional samples. The same
individual observation is not followed across windows; this is why the field
licenses distribution-level transport only.

## TorchSig

TorchSig inputs are generated synthetic signals and frozen derived arrays. The
Round 11C panel preserves latent-signal identity across deployment windows and
shares the latent random draws between baseline and Cliff-aware model arms.
This pairing is a controlled experimental construction, not personal or
restricted data.

Round 12C uses independent generated acquisition and evaluation panels for
each training seed. Its acquisition labels are revealed only after the common
query pool is fixed. The 16-cell coverage map uses queried true class, declared
deployment path, and early/late phase; it contains no physical impairment
coordinate.

## CIFAR-10, CIFAR-10-C, and CURE-TSR

Source CIFAR images are intentionally excluded. The merged Round 13 directory
contains model checkpoints, per-identity predictions and true margins, labels,
corruption names, severity levels, repair selections, protocols, and
machine-readable summaries. These derived files are sufficient to audit the
reported paired transition accounting and repair comparisons.

- CIFAR-10-C record: https://doi.org/10.5281/zenodo.2535967
- expected `CIFAR-10-C.tar` MD5:
  `56bf5dcef84df0e2308c6dcbcbbd8499`
- expected CIFAR-10 Python archive MD5:
  `c58f30108f718f92721af3b95e74349a`
- segmented download and checksum scripts:
  `round13_second_domain/src/download_cifar10_segmented.sh` and
  `round13_second_domain/src/download_cifar10c_segmented.sh`

CURE-TSR data are not included. Round 13A records `DATA_UNAVAILABLE`; public
metadata did not establish paired identity across severity and no CURE-TSR
result is claimed.

## CURE-OR

CURE-OR is distinct from CURE-TSR. The merged `cure_or/` directory contains
public mini CURE-OR metadata tables, frozen identity/stream/training/repair
tables, derived ConvNeXt features, blind predictions, Hybrid25 telemetry,
calibration decisions, all Phase 2 raw result tables, source code, protocols,
and integrity records. It does not include the multi-gigabyte source image
archives or pretrained ConvNeXt weights.

- CURE-OR paper: https://doi.org/10.1109/ICMLA.2018.00028
- mini CURE-OR v2 data record: https://doi.org/10.5281/zenodo.4299330
- integrated experiment ZIP SHA-256:
  `e3c3508eb800859baace03f6d1259cde6c4d49fdc453a902fc91d61bd196d143`
- expected ConvNeXt-Tiny weight SHA-256:
  `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`

The frozen split keeps 50 class-balanced calibration identities and 50
disjoint class-balanced confirmation identities. The Phase 1 commitment fixes
all confirmation predictions, telemetry, alarms, and repair decisions without
scoring confirmation labels. The model seed is the inferential unit; schedule,
family, window, and identity observations are not independent replicates.

## Privacy and redistribution

The artifact contains no personal data. Before public redistribution, retain
the upstream Covertype, TorchSig, CIFAR-10, CIFAR-10-C, and CURE-OR attributions and
replace the manuscript's temporary artifact sentence with a permanent
repository and archival DOI.
