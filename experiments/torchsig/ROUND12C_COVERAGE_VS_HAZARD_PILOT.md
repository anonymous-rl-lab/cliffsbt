# Round 12C: deployment coverage versus hazard concentration

## Result

The frozen five-seed pilot supports the proposed mechanism in the tested
TorchSig small-budget regime:

> Repair efficacy is primarily governed by coverage of the observed deployment
> domain, not by concentrating the budget on samples with the largest local
> decision-boundary hazard.

All 14 pretarget gates passed. The decision is
`PILOT_SUPPORTS_COVERAGE_DOMINANCE`.

## Direct manipulation check

Every repair arm replaced 16 of 320 training examples (5%) and used the same
queried pool within a training seed.

| Arm | Mean occupied-cell fraction | Mean local hazard score | True local-incident precision | Mean terminal-risk reduction |
|---|---:|---:|---:|---:|
| Random unstratified | 0.5875 | 1.0478 | 0.1625 | 0.2188 |
| Hazard concentrated | 0.4375 | 4.8575 | 0.7875 | 0.1520 |
| Coverage random | 1.0000 | 0.5240 | 0.0750 | 0.2461 |
| Coverage + hazard | 1.0000 | 2.5692 | 0.4125 | 0.2465 |

The manipulation was strong in the direction required for falsification:
hazard concentration raised the selected hazard score by 4.3336 and the true
incident hit rate by 71.25 percentage points relative to coverage-random, while
covering only 43.75% rather than 100% of the 16 deployment cells.

## Primary paired result

The predeclared primary contrast was the seed-paired difference in terminal-risk
reduction:

`coverage_random - hazard_concentrated`.

- Mean: **0.09414**.
- Training-seed cluster bootstrap 95% CI: **[0.03633, 0.14648]**.
- Positive seeds: **4/5**.
- Seed contrasts: 0.13867, 0.08008, 0.17773, 0.08984, -0.01563.
- Median coverage-random reductions were positive on both paths: 0.24609 on
  noise and 0.21875 on mixed-gradient.

The baseline crossed the common relative boundary on 10/10 seed-path pairs.
Maximum baseline-distribution validation loss across all repair models was
0.0000.

## Factorial interpretation

Adding coverage while starting from the hazard-concentrated arm improved mean
terminal-risk reduction by **0.09453**. Adding hazard ranking after full
coverage improved it by only **0.00039**; its post hoc seed-cluster bootstrap
interval spans zero. Therefore hazard is useful for identifying mechanism
activity, but it does not supply measurable additional repair value once the
small repair set covers the deployment cells.

Coverage-random exceeded unstratified random by 0.02734 on average, but the
five-seed interval spans zero. The licensed claim is therefore the direct
coverage-versus-hazard result, not universal superiority over every random
sampling realization.

## Claim boundary

“Coverage” is operationalized here as 16 cells:

`queried true class (4) x deployment path (2) x early/late phase (2)`.

The result does not prove that these axes are universally sufficient, does not
separate the contribution of each axis, and does not establish a theorem for
other architectures or domains. It establishes that, under this frozen 5%
TorchSig intervention, a diverse set spanning the observed deployment field is
more valuable than repeatedly hitting the highest-hazard local boundary
fragments.

Round 12C is a fresh post-paper pilot. It is not evidence in Paper V5.

## Reproducibility

- Pretarget digest: `853e7cc450db493cd8e4bad4079bff1758e14e1fb0ee482b822927cde6b32090`.
- Runtime: TorchSig v2.1.1 official-source NumPy execution, source commit
  `d9abfe1af2b0216d2bacc31c677407ed31878086`.
- Two pre-evaluation technical aborts are retained in the result directory.
  Neither generated an evaluation stream or changed any scientific design
  element.
- New mechanism tests: 5/5 passed.
- Full repository tests under the disclosed source-faithful runtime: 38/38
  passed.
- Per-file hashes: `results/round12c_coverage_vs_hazard_pilot/SHA256SUMS.txt`.
