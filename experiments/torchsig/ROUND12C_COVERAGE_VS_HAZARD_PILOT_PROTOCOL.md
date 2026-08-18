# Round 12C pretarget protocol: deployment coverage versus hazard concentration

## Question

This pilot tests one narrow claim:

> Under a fixed 5% repair budget, deployment-domain coverage is the dominant driver of repair, rather than the hit rate on individually hazardous samples.

The experiment is a post-paper mechanism pilot. It is not evidence in Paper V5 unless a later manuscript revision explicitly promotes it.

## Frozen design

- Five fresh training seeds are used. Each seed has its own acquisition and evaluation streams.
- Every arm starts from the same 320-example training set for that seed and replaces exactly 16 examples (5%). Class counts, model architecture, tree count, and model random state are paired.
- The acquisition pool is common to all arms. It is selected without outcome labels using equal quotas over deployment path and anchor-predicted class. Labels are revealed only after this common query set is frozen.
- The deployment domain is partitioned into 16 cells:
  `true class (4) x deployment path (2) x phase (early/late)`.
- Early is windows 7--13 inclusive; late is windows 14--20 inclusive. Window 6 is the anchor.
- A candidate's hazard score is computed only after its label is queried. It combines persistent local crossing, post-crossing persistence, margin descent, and negative margin area within the phase band.
- Acquisition and evaluation streams are disjoint. All models within a seed face the exact same evaluation samples.

## Arms

1. `random_unstratified`: 16 random queried trajectories, no deployment-cell quota.
2. `hazard_concentrated`: the 16 largest queried local-hazard scores globally, no deployment-cell quota.
3. `coverage_random`: one randomly chosen queried trajectory from every deployment cell.
4. `coverage_hazard`: one highest-hazard queried trajectory from every deployment cell.

Every selected training example is the observed state at its chosen phase-local repair time. A source trajectory may enter an arm only once.

## Primary estimand

For each seed, average the two fresh deployment paths and compute baseline minus repaired terminal risk. The primary paired contrast is

`coverage_random reduction - hazard_concentrated reduction`.

The claim is supported only if all of the following pretarget gates pass:

1. `hazard_concentrated` has higher selected mean hazard score than `coverage_random`.
2. `coverage_random` occupies all 16 cells and has higher coverage than `hazard_concentrated`.
3. Baseline common-boundary cliff fraction is at least 0.80.
4. Mean primary contrast is at least 0.01.
5. At least four of five seed contrasts are positive.
6. The seed-cluster bootstrap 95% interval for the primary contrast has lower endpoint above zero.
7. Adding coverage at high hazard (`coverage_hazard - hazard_concentrated`) improves terminal-risk reduction.
8. The coverage contribution exceeds the within-coverage hazard contribution.
9. `coverage_random` has positive median terminal-risk reduction on both deployment paths.
10. Maximum validation-accuracy loss across repair arms is at most 0.02.

Incident-crossing and risk-area results are frozen secondary endpoints. They cannot rescue a failed primary terminal-risk test.

## Decision

- All primary gates pass: `PILOT_SUPPORTS_COVERAGE_DOMINANCE`.
- Any primary gate fails: `PILOT_DOES_NOT_SUPPORT_COVERAGE_DOMINANCE`.
- Acquisition cannot populate every cell before evaluation: `PRETARGET_ACQUISITION_ABORT`.

No seed replacement, gate change, band change, or selector tuning is allowed after the pretarget digest is frozen.
