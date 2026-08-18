# Covertype 45 m signed-boundary formal confirmation v1

## Licensed hypothesis

The bounded multi-seed pilot passed all nine frozen gates on a panel known-disjoint from
the preceding probability-flow probes. This confirmation asks whether the result
repeats under both a new training panel and a new, larger evaluation panel.

The 45 m scale was originally selected after a Covertype scale diagnostic. Therefore
this experiment formally confirms the signed-boundary mechanism conditional on the
45 m protocol; it is not an unbiased discovery of the scale itself.

## Independent panel relative to the pilot

- Thirteen 45 m blocks cover 2745--3330 m.
- Before selection, all indices from the original source smoke, first boundary-flow
  panel, large accounting panel, and 45 m multi-seed pilot are excluded.
- Every block contributes 96 new fit examples and 256 new evaluation examples per
  class. Fit and evaluation indices are globally unique.
- The smallest remaining block/class pool contains 440 examples, exceeding the frozen
  requirement of 352.

## Frozen models and paths

Five new MLP seeds are used. Ascending and descending paths start at block positions 3,
7, and 10 and contain three target blocks. This yields 30 independently fitted models
and 60 paired 45 m transitions.

## Statistic and controls

The statistic is class-conditioned signed boundary hazard: deployment displacement in
the final hidden representation projected onto the frozen model's signed final-layer
normal, multiplied by mass in a training-fixed margin band.

Controls are unsigned hidden-space distance change, normal speed without crowding, and
128 coordinate-permuted final-layer normals. Outcomes are Error and Brier changes.
Uncertainty uses a two-way source/model-seed cluster bootstrap.

## Frozen gates

The nine pilot gates are reused unchanged:

1. hazard/Error Spearman >= 0.50;
2. hazard/Brier Spearman >= 0.50;
3. error-change sign accuracy >= 0.75;
4. hazard advantage over unsigned distance >= 0.10;
5. at least four of six sources have sign accuracy >= 0.60;
6. both direction-specific correlations are positive;
7. true-boundary correlation exceeds the placebo 95th percentile;
8. two-way bootstrap hazard-correlation lower bound > 0;
9. two-way bootstrap advantage lower bound > 0.

All gates must pass. No source, seed, direction, transition, metric, or control may be
removed after target inspection.
