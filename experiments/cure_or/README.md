# CURE-OR v2 complete reproducibility package — c6ygf

This package consolidates the complete authoritative CURE-OR v2 experiment:
frozen source code and configuration, official dataset metadata, frozen input
tables, Phase 1 outcome-blind arrays, Phase 2 raw result tables, audit reports,
and the final formation–warning–repair conclusion.

The machine decision is:

`FORMATION_WARNING_REPAIR_CONFIRMED`

## Authoritative evidence chain

- Frozen registration: <https://osf.io/c6ygf>
- Phase 1 blind commitment in the associated project: <https://osf.io/nm3ex/files/x8vmb>
- Phase 2 confirmatory evidence: <https://osf.io/nj3hp>

Exact archive and commitment hashes are recorded in `PROVENANCE.json`.

## Package map

- `code/`: acquisition, feature extraction, Phase 1, Phase 2, tests, manifest,
  and independent package audit.
- `config/`: frozen experiment, Hybrid25, warning, repair, schema, and
  environment specifications.
- `data/`: official mini CURE-OR metadata/CSVs and every frozen identity,
  stream, schedule, training, and repair-candidate table.
- `raw_outputs/`: feature tensors, predictions, telemetry, alarms, calibration
  decisions, retrieval/CRC evidence, and all path/seed/repair result tables.
- `audit/`: Phase 1 and Phase 2 execution notes, logs, and audit reports.
- `docs/EXPERIMENT_PROTOCOL.md`: complete experimental design and gates.
- `docs/DATA_DICTIONARY.md`: input and output definitions, axes, and shapes.
- `docs/RESULTS_AND_CONCLUSIONS.md`: complete numerical results and bounded
  scientific conclusions.
- `docs/REPRODUCIBILITY.md`: audit, re-scoring, and full rerun instructions.
- `PACKAGE_MANIFEST.sha256`: non-circular SHA-256 manifest.
- `PACKAGE_AUDIT.json`: machine-readable final audit.

## Fast verification

From the package root:

```bash
PYTHONPATH=code python code/audit_package.py --strict
```

The audit verifies every manifest entry, the OSF Phase 1 blind commitment,
all raw evidence manifests, numerical gates, array axes/shapes, table row counts,
the frozen design, and absence of retired experimental material.

## Data boundary

The package includes the official `train.csv`, `test.csv`, Zenodo metadata,
frozen image assignments, retrieval/CRC manifest, and all derived numerical
features and predictions. It does not redistribute the approximately 3.4 GB
mini CURE-OR image archives or ConvNeXt-Tiny weights. Their canonical locations,
licenses, sizes, checksums, and the exact weight SHA-256 are included.

## Confirmatory boundary

The inferential unit is the classifier-head model seed (`n = 5`). The 150
model–schedule–family paths are operational paths, not independent replicates.
The result is a same-benchmark prospective confirmation over fixed model seeds
and schedule variants. It is not a fresh-identity, cross-dataset, cross-domain,
or natural longitudinal-drift replication.
