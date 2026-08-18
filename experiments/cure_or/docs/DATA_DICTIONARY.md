# Data dictionary

## Frozen inputs

| Path | Definition |
|---|---|
| `data/train.csv` | Official mini CURE-OR training metadata. |
| `data/test.csv` | Official mini CURE-OR test metadata. |
| `data/record.json` | Canonical Zenodo record metadata and source-file checksums. |
| `data/DATASET_MANIFEST.json` | Dataset row/count audits and local/canonical checksums. |
| `data/DATA_SPLIT_FROZEN.csv` | Fixed calibration/confirmation identity assignment. |
| `data/TARGET_STREAMS_FROZEN.json` | Exact baseline and severity 1–4 image IDs for every family/identity stream. |
| `data/TRAINING_BASELINE_FROZEN.csv` | Fixed 150-image clean head-training pool. |
| `data/REPAIR_CANDIDATES_FROZEN.csv` | Fixed repair candidates and fragment/rank fields. |
| `data/FAMILY_ASSIGNMENT.json` | Ten target families and their valid baselines/severity levels. |
| `data/MODEL_SEEDS.json` | Five registered classifier-head seeds. |
| `data/SCHEDULE_IDS.json` | Three frozen asynchronous schedule identifiers. |

The identity key is `(class, background, perspective)`. Every stream follows
the same base identity through its challenge severities.

## Phase 1 raw numerical objects

### `raw_outputs/features.npz`

| Array | Shape | Type | Meaning |
|---|---:|---|---|
| `test_ids` | `(4200,)` | `int32` | Ordered frozen test image IDs. |
| `test_features` | `(4200, 768)` | `float32` | Frozen ConvNeXt test representations. |
| `train_ids` | `(1650,)` | `int32` | Ordered frozen training image IDs. |
| `train_features_seed113` | `(1650, 768)` | `float32` | Seed-specific training representations. |
| `train_features_seed127` | `(1650, 768)` | `float32` | Seed-specific training representations. |
| `train_features_seed139` | `(1650, 768)` | `float32` | Seed-specific training representations. |
| `train_features_seed151` | `(1650, 768)` | `float32` | Seed-specific training representations. |
| `train_features_seed163` | `(1650, 768)` | `float32` | Seed-specific training representations. |

### `raw_outputs/blind_predictions.npz`

The main axis order is model seed × schedule × family × window × identity.

| Array | Shape | Meaning |
|---|---:|---|
| `baseline_predicted` | `(5, 3, 10, 13, 50)` | Baseline-head class predictions. |
| `candidate_repair_predicted` | `(5, 3, 10, 13, 50)` | Candidate repaired-head predictions. |
| `deployed_repair_predicted` | `(5, 3, 10, 13, 50)` | Safety-gated deployed predictions. |
| `hybrid25_sensors` | `(5, 3, 10, 13, 25)` | Committed outcome-blind telemetry. |
| `warning_scores` | `(5, 3, 10, 13)` | Frozen Hybrid25 scores. |
| `seeds` | `(5,)` | Model-seed axis labels. |
| `schedule_ids` | `(3,)` | Schedule axis labels. |
| `families` | `(10,)` | Challenge-family axis labels. |

Additional Phase 1 files:

- `fetch_manifest.json`: image member retrieval, size, CRC, and cache evidence.
- `calibration_assessment.json`: allowed calibration warning summary, repair
  decisions, and selected repair image IDs.
- `blind_alarm_rows.json`: complete Hybrid25 score series and first alarms for
  all 150 paths.
- `run_metadata.json`: array ordering, hashes, environment binding, and
  outcome-blind declaration.
- `BLIND_OUTPUTS.sha256`: blind commitment over the Phase 1 decision objects.

## Phase 2 raw tables

### `path_level_results.csv` — 150 rows

One row per model seed × schedule × family. It records eligibility, baseline
and endpoint risk, the 13-window risk series, cliff event, alarm, timely/false
classification, forward/recovery transitions, residual series, and maximum
closure error.

### `repair_path_results.csv` — 150 rows

One row per model seed × schedule × family. It records repair eligibility,
baseline and deployed risk series, event times, mean/endpoint risk gains, and
event-time gain.

### `seed_level_results.csv` — 5 rows

One row per inferential model seed. It records warning counts/rates/lead,
repair eligibility, mean risk gain, and mean event-time gain after pooling that
seed's schedules and families.

### `results.json`

Machine-readable H1/H2/H3 gates, pooled and per-seed results, integrity state,
claim boundary, and final decision.

`POSTREVEAL.sha256` binds the Phase 2 tables. `EVIDENCE.sha256` binds the blind
objects, blind manifest, Phase 2 tables, and post-reveal manifest.
