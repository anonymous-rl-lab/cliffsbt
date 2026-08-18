# Data specification — authoritative v2 package

## Canonical source

mini CURE-OR, Zenodo record 4299330, DOI `10.5281/zenodo.4299330`,
CC-BY-4.0.

Official files:

| file | bytes | canonical checksum |
|---|---:|---|
| train.zip | 1,883,200,293 | MD5 c370f94d0ba9f90f9abc75a5d1a2aca5 |
| test.zip | 1,526,714,682 | MD5 b495d36b5df29b18584d01c25c0bfefd |
| train.csv | 153,396 | MD5 a2489e30a7e01fc022d0c4924de9f5e6 |
| test.csv | 109,526 | MD5 4c0c7330fd8f312d1ccacbac1f2a288a |

The package copies the two official CSV files and the Zenodo record metadata.
The 3.4 GB source image archives are not redistributed. `code/data_access.py`
retrieves only the image IDs listed in the frozen manifests and verifies the CRC
and uncompressed size recorded in each canonical ZIP central directory.

## Frozen tables

- `FAMILY_ASSIGNMENT.json`: target families, names, and family-appropriate
  baselines.
- `DATA_SPLIT_FROZEN.csv`: every test identity assigned to calibration or
  confirmation and, for confirmation, a balanced panel.
- `TARGET_STREAMS_FROZEN.json`: exact baseline and level 1--4 image IDs for every
  prospective identity-family stream.
- `MODEL_SEEDS.json`: the five registered v2 classifier-head seeds.
- `SCHEDULE_IDS.json`: the three frozen v2 asynchronous schedule identifiers.
- `TRAINING_BASELINE_FROZEN.csv`: the 150 clean training images.
- `REPAIR_CANDIDATES_FROZEN.csv`: every level-3 training candidate in the
  prospective fragment pool and its fixed dense-floor rank.
- `DATASET_MANIFEST.json`: official and local checksums plus row/count audits.

The identity tables are fixed from public metadata and stable hashes. V2 does
not claim new identities, fresh data, cross-dataset replication, or cross-domain
replication. The prospective units are the registered classifier-head seeds and
their schedule-specific outputs.

## Raw experimental outputs

The package-level `raw_outputs/` directory contains the complete recorded
feature tensors, prediction arrays, telemetry, calibration decisions, image
retrieval/CRC manifest, and Phase 2 result tables. `features.npz` therefore
provides the derived numerical input needed to audit or re-score the experiment
without redistributing the copyrighted image archives.

## Identity key

The base identity is `(class, background, perspective)`. The same base is followed
through every severity in a family. Image IDs differ by challenge condition, but
the base identity does not.

## Challenge baselines

Color families 2 and 6 start at challenge type 1 (no challenge). Grayscale
families 11--18 start at challenge type 10 (grayscale). This preserves an ordered
within-modality path and prevents grayscale conversion itself from being treated
as a severity transition.
