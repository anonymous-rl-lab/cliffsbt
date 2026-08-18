# Lightweight reduction and integration audit

## Source packages

| Source | SHA-256 | Size |
|---|---|---:|
| Full four-domain evidence archive | `751e2b1f5a10e6bd1201087475742cd338f6e4f647030db2ab83af33ab8c73e5` | 151,026,589 bytes |
| Manuscript v7 package | `169ab40834440659eebaba1af657333d0c79eef57d3bc7d223a513a6dad4f142` | 3,389,822 bytes |
| `sbt-monitor` 0.1.0 source | `21abee3adb68d803ab2860c0bff355115ad19a7d7b1603265750683eb4a43104` | 59,527 bytes |

## Cleaning result

The GitHub tree remains approximately 8.5 MB before ZIP compression, a **94.4% reduction** relative to the full compressed evidence archive while adding the v7 manuscript, final figures, compact figure inputs and the reusable `sbt-monitor` source.

### Removed from GitHub

- source image archives and public-dataset caches;
- CURE-OR `features.npz` and other large intermediate tensors;
- model checkpoints and duplicate raw prediction arrays;
- repeated historical figures and target arrays;
- probe-stage result directories not required for the final compact route;
- Python caches, virtual environments and machine-local paths.

### Retained

- formal experiment source, frozen configurations and protocols for four domains;
- compact committed evidence sufficient for manuscript-number verification and exact figure regeneration;
- v7 manuscript, SI, editorial abstract, reference figures and figure inventory;
- `sbt-monitor` source, tests and scientific-scope specification;
- complete full-archive SHA-256 bridge and provenance documents;
- negative-result and STOP boundaries in the retained protocols/docs.

The compact route is a verification/reproduction layer, not a claim that model retraining can be performed without the omitted public source data and weights.


## CI tracking hardening

Patch 1.0.1 explicitly tracks the frozen CURE-OR Phase 1 execution log, verifies every manifest entry against the Git index before reproduction, and disables matrix fail-fast so Python 3.10 and 3.12 report independently.
