# Data notice

The source CIFAR-10 and CIFAR-10-C image archives are not redistributed in the
Round 13 evidence package.

- CIFAR-10-C record: https://doi.org/10.5281/zenodo.2535967
- expected archive filename: `CIFAR-10-C.tar`
- expected MD5: `56bf5dcef84df0e2308c6dcbcbbd8499`
- local extracted directory expected by the evaluator: `data/CIFAR-10-C/`
- CIFAR-10 archive filename: `cifar-10-python.tar.gz`
- CIFAR-10 expected MD5: `c58f30108f718f92721af3b95e74349a`
- CIFAR-10 extracted directory expected by the retrainer:
  `data/cifar-10-batches-py/`

The package includes model checkpoints, per-identity predictions and true
margins, labels, corruption names, severity levels, frozen protocols, and
machine-readable summaries. Those derived outputs are sufficient to audit the
reported transition accounting without redistributing the source images.

CURE-TSR is not included. Its official repository requires users to request a
download link and accept the dataset conditions. Round 13A therefore records
`DATA_UNAVAILABLE`, and no CURE-TSR result is claimed.
