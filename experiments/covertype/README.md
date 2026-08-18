# Cliff–Covertype mechanism package v1

This package closes the Covertype field investigation of Cliff. It preserves the full
positive and negative evidence chain rather than presenting only the successful probes.

## Scientific outcome

Covertype supports a **distribution-level, model-conditional decision-boundary
transport mechanism**:

1. unsigned displacement from training is not a sufficient risk coordinate;
2. 15 m window-to-window derivatives are below the reproducible change scale;
3. 45 m is an empirical coherence scale for this protocol;
4. global mean margin velocity is not stable across training–deployment panel pairs;
5. the proposed shoulder-local synchronization statistic fails its controls and is
   rejected;
6. boundary density multiplied by robust median equal-quantile margin velocity
   reproduces strongly on the last fresh holdout, but one of ten frozen gates fails.

The terminal status is therefore **strong partial external mechanism support**, not a
complete mechanism proof and not an all-gates pass.

## Read first

- `SCIENTIFIC_SYNTHESIS_ZH.md`: complete Chinese scientific synthesis.
- `SCIENTIFIC_SYNTHESIS_EN.md`: concise manuscript-facing English synthesis.
- `EXPERIMENT_LEDGER.md`: chronological experiment and decision ledger.
- `CLAIM_BOUNDARIES.md`: claims that are and are not licensed.
- `REPRODUCIBILITY.md`: environment, data, and rerun instructions.
- `DATA_NOTICE.md`: source attribution, license, hashes, and panel semantics.
- `integrity/MANIFEST.sha256`: file-level integrity manifest.
- `integrity/PACKAGE_AUDIT.json`: machine-readable final audit.

## Directory map

- `work_covtype_probe/`: original source-applicability smoke and 5% support-intervention
  pilot, including raw UCI data.
- `work_covtype_boundary_flow_probe/`: 15 m flow forecast, reliability diagnostics,
  scale audit, and one-seed 45 m accounting.
- `work_covtype_45m_multiseed_pilot/`: known-disjoint 45 m five-seed pilot; all gates
  passed.
- `work_covtype_45m_formal_v1/`: new-fit/new-evaluation formal confirmation; stopped.
- `work_covtype_local_flow_probe/`: rejected shoulder-local synchronization probe and
  robust margin-transport diagnostics.
- `work_covtype_robust_margin_holdout/`: last fresh target holdout; 9/10 gates.
- `sklearn_data/covertype/`: exact local scikit-learn source cache used by later runs.
- `evidence/`: preserved applicability report and historical C1/C2 ledger.

## Terminal interpretation

The strongest supported statistic is

```text
boundary density × (− median equal-quantile signed-margin velocity)
```

On the final fresh holdout it reaches Error/Brier Spearman 0.746/0.821, while unsigned
training distance is −0.029 and the 95th percentile of permuted boundaries is 0.269.
The two-way cluster interval for the robust statistic is [0.485, 0.879]. Its point
advantage over ordinary mean transport is +0.314, but that advantage interval is
[−0.052, 0.678]; therefore strict superiority over the mean is not formally certified.

Covertype has no longitudinal sample identity. It cannot establish literal sample-level
crossing synchrony; that question must be tested in a paired degradation system such as
TorchSig.
