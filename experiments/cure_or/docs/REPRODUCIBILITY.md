# Reproducibility instructions

## Environment

The formal execution used:

- Python 3.12.13
- NumPy 2.5.2
- PyTorch 2.13.0+cpu
- torchvision 0.28.0+cpu
- Pillow 12.3.0
- scikit-learn 1.9.0
- CPU float32 ConvNeXt extraction
- NumPy float64 head fitting and scoring

The lock file is `environment.lock`.

## 1. Audit the delivered package and results

From the package root:

```bash
PYTHONPATH=code python code/audit_package.py --strict
```

This is read-only unless `--write-report` is explicitly added. The delivered
`PACKAGE_AUDIT.json` records the release audit.

## 2. Re-score Phase 2 from the committed Phase 1 objects

Copy `raw_outputs/` to a disposable workspace so the delivered evidence remains
unchanged:

```bash
cp -a raw_outputs /tmp/cure_or_v2_rescore
PYTHONPATH=code python code/run_phase2.py \
  --workspace /tmp/cure_or_v2_rescore
```

The command verifies the blind commitment, reads confirmation truth from the
frozen stream tables, reconstructs all H1/H2/H3 results, and writes the four
Phase 2 result objects plus their manifests.

## 3. Full rerun from canonical images

Obtain the official ConvNeXt-Tiny weight file whose SHA-256 is:

`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.

Then run Phase 1 in a new workspace:

```bash
PYTHONPATH=code python code/run_phase1.py \
  --workspace /absolute/path/CURE_OR_V2_RERUN \
  --data-cache /absolute/path/cure_or_image_cache \
  --weights /absolute/path/convnext_tiny-983f1562.pth \
  --action all \
  --workers 16
```

The acquisition code retrieves only frozen image members from the canonical
mini CURE-OR archives and checks member size and CRC. After Phase 1 completes,
verify that `confirmation_labels_scored` is false and archive the committed
workspace before executing Phase 2:

```bash
PYTHONPATH=code python code/run_phase2.py \
  --workspace /absolute/path/CURE_OR_V2_RERUN
```

Never reuse a workspace that already contains `BLIND_OUTPUTS.sha256`. The Phase
1 code refuses to overwrite a blind commitment.

## 4. Primary files for independent analysis

- Use `raw_outputs/path_level_results.csv` for formation and warning path
  analysis.
- Use `raw_outputs/repair_path_results.csv` for paired repair trajectories.
- Use `raw_outputs/seed_level_results.csv` for the five inferential clusters.
- Use `raw_outputs/blind_predictions.npz` to reconstruct outcomes from committed
  prediction arrays.
- Use `raw_outputs/features.npz` to audit the frozen numerical representation
  without downloading source images.
- Use `raw_outputs/results.json` as the normative machine-readable result.

Any alternative threshold, sensor, exclusion, bootstrap unit, or repair policy
is exploratory and must not replace the registered confirmatory result.
