# Round 13E | Coverage versus hazard-hit repair

Frozen after Round 13D and before any Round 13E model was trained or evaluated.

## Scientific question

Under an equal small labeled repair budget, is deployment-fragment coverage more
valuable than repeatedly selecting the most severe observed boundary-crossing
examples?

## Leakage control and common stream

- Official CIFAR-10-C identities are split once, stratified by class.
- Calibration identities: 2,000; formal holdout identities: 8,000.
- Selection uses only calibration identities and the frozen Round 13D outputs.
- All arms and all new training seeds face the same holdout identities, 15
  corruption families, and five severities.
- The 20,000 clean training identities are common to every arm.
- Operational thresholds are fixed per new seed from its no-repair clean error
  plus 0.15, then reused unchanged for all repair arms of that seed.

## Equal-budget arms

Formal budget is 1,000 labeled corrupted images, equal to 5% of the common clean
training set.

1. `baseline`: clean training set only.
2. `random`: 1,000 uniformly sampled calibration corruption cells.
3. `hazard`: the 1,000 deepest ensemble-confirmed first-crossing examples,
   globally ranked without a coverage constraint.
4. `coverage`: the same first-crossing candidate population, sampled by
   round-robin coverage over corruption-family x class x first-crossing-severity
   fragments, with hazard score used only within a fragment.

Hazard and coverage must both have first-crossing hit rate 1.0. This matches
dangerous-sample hit rate and isolates setwise coverage.

## Selector and retraining

- Selector: frozen ensemble of the Round 13D seeds 31, 47, and 61.
- New retraining seeds: 71, 83, and 97.
- Architecture: frozen Round 13 SmallCNN.
- Common clean subset: 20,000 identities, seed 2026.
- Training: from scratch, 8 epochs, AdamW, common optimizer schedule.
- Each completed seed-arm fit immediately writes a checkpoint, raw paired
  outputs, metrics, and resumable state.

## Pretarget formal endpoints

- mean endpoint error across 15 corruption families;
- mean risk area across all five corrupted severities;
- fraction of corruption families exhausting the common operational threshold;
- mean cumulative endpoint net flux;
- clean error;
- task-boundary accounting error.

Training seed is the uncertainty cluster. Paired seed-cluster bootstrap uses
20,000 draws.

## Frozen formal gates

| Gate | Criterion |
|---|---|
| Equal dangerous-hit rate | hazard = coverage = 1.0 and both budgets = 1,000 |
| Coverage separation | coverage fragments >= 2 x hazard fragments and >=14 families |
| Model competence | clean accuracy >=0.45 for every seed-arm |
| Exact paired accounting | maximum absolute error <=1e-12 |
| Coverage beats hazard at endpoint | upper 95% CI of coverage - hazard <0 |
| Coverage beats hazard on risk area | upper 95% CI of coverage - hazard <0 |
| Coverage reduces crossing prevalence | mean coverage - hazard <0 |
| Coverage improves no-repair endpoint | upper 95% CI of coverage - baseline <0 |
| Clean-risk guard | upper 95% CI of coverage - hazard clean error <=0.02 |

All nine gates are required for the strong claim. A partial result is retained
without changing the protocol.

## Smoke authorization

Before formal execution, one engineering seed (67), 5,000 clean examples, budget 250,
three epochs, 500 calibration identities, and 1,000 holdout identities are used
only for engineering gates: exact budgets, matched hazard hit rate, greater
coverage, finite metrics, clean accuracy >=0.30, and exact accounting.
Superiority is not a smoke gate. Formal execution begins only if all engineering
gates pass; formal seeds 71, 83, and 97 remain unrevealed.
