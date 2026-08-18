# Data notice and attribution

## Source

Blackard, J. (1998). *Covertype* [Dataset]. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C50K5N

Official record: https://archive.ics.uci.edu/dataset/31/covertype

The official repository describes 581,012 instances, 54 predictors, seven cover types,
and no missing values. This package uses the binary Cover Type 1 versus Cover Type 2
task and balanced class panels along elevation.

## License

The UCI Covertype page distributes the dataset under Creative Commons Attribution 4.0
International (CC BY 4.0). The raw archive is included with attribution.

## Exact source files

- `work_covtype_probe/data/covtype.data.gz`
  - SHA-256: `614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`
- `sklearn_data/covertype/samples_py3`
  - SHA-256: `ba15c20add4a83550488f189a3c4b1d774167e84256d99844d5327f4174323ff`
- `sklearn_data/covertype/targets_py3`
  - SHA-256: `9c41047c16f75a49a2e6852913a504a653892a856ffe8698ba918901a550eee5`

The raw gzip archive was downloaded from:

`https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz`

## Derived data

All `.npz`/`.npy` files in experiment `data/` directories are fixed integer index
panels, not altered feature matrices. All CSV/JSON result files are derived summaries
or per-transition ledgers. Later panels record the union of known earlier indices and
were audited for zero known overlap.

The historical C1/C2 arrays were unavailable after workspace remount; only their exact
session ledger is included under `evidence/`.
