# Reproducibility tiers

## Tier 1 — compact committed evidence

Runs in minutes on CPU and requires no benchmark downloads:

```bash
python reproduce/verify_compact_evidence.py
python reproduce/make_figures.py --evidence-dir evidence/compact --out-dir figures/rebuilt
```

Compact evidence was deterministically derived from:

- full evidence archive SHA-256 `751e2b1f5a10e6bd1201087475742cd338f6e4f647030db2ab83af33ab8c73e5`;
- manuscript v7 package SHA-256 `169ab40834440659eebaba1af657333d0c79eef57d3bc7d223a513a6dad4f142`;
- `sbt-monitor` source archive SHA-256 `21abee3adb68d803ab2860c0bff355115ad19a7d7b1603265750683eb4a43104`.

## Tier 2 — domain reruns

Each domain has separate dependencies and must not be forced into one environment.

- `experiments/torchsig`: formal warning, training intervention, paired transport and sparse repair scripts.
- `experiments/covertype`: final 45 m formal panel and fresh robust holdout, with frozen index panels but without the large sklearn cache.
- `experiments/cifar10c`: CIFAR-10/CIFAR-10-C download scripts, paired mechanism and repair code; source image archives/checkpoints are omitted.
- `experiments/cure_or`: registered phase code, frozen metadata/splits/configuration; source images, ConvNeXt weights and `features.npz` are omitted.

Read each domain README and protocol before an intentional rerun. Frozen STOPs must remain STOPs.

## Tier 3 — full evidence archive

The 145 MB archive is not duplicated in GitHub. Its hash and original internal SHA256 list are retained under `evidence/`. After archival deposition:

```bash
python reproduce/download_full_evidence.py --url <ARCHIVE_URL>
```

The download is accepted only if its SHA-256 equals the frozen value.

## Figure reproduction

The committed reference figures are under `figures/reference`. The compact figure builder reads only `evidence/compact`; it does not silently depend on machine-local absolute paths, private caches, or external network access.

## Rebuilding post hoc diagnostic tables

The original v6 and v7 diagnostic builders are retained in `reproduce/diagnostics/`. They require the complete CURE-OR/CIFAR committed arrays from the full evidence archive. Their outputs are already included in `evidence/compact/v6_diagnostics/` and `v7_diagnostics/` for the lightweight route.
