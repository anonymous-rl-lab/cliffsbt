# Covertype 45 m signed-boundary multi-seed pilot v1

## Motivation and status

The 15 m C2 experiment measured risk levels reliably but measured local risk changes
and distribution velocities unreliably. A posttarget diagnostic identified 45 m as the
smallest tested scale with reproducible path direction and risk-trend sign. A one-seed
45 m accounting smoke produced model-specific signal but did not pass its complete
gate. This pilot tests whether that signal survives model randomness and a new known-
disjoint evaluation panel.

Because the 45 m scale was selected after inspecting earlier Covertype outcomes, this
is a bounded validation pilot, not final independent evidence.

## New block panel

- The usable interior is represented by 13 non-overlapping 45 m blocks over
  2745--3330 m. The 2700--2745 m edge block is excluded pretarget because only 120
  known-unused minority-class samples remained, insufficient for disjoint fitting and
  evaluation panels.
- All indices in the original source-smoke panel, first boundary-flow panel, and large
  accounting panel are excluded before selection.
- Each block contributes 96 fit examples and 128 disjoint evaluation examples per
  class.
- The new fit and evaluation indices are globally unique and known-disjoint from all
  locally available prior Covertype probe panels.

## Frozen paths and models

The same conceptual early/middle/late paths are represented by origin blocks 3, 7, and
10 in both ascending and descending directions. Each target contains three blocks,
yielding two 45 m transitions per source. Five new MLP seeds are fitted, for 30 models
and 60 source-seed transitions.

## Mechanism statistic

For every transition, class-conditional final-hidden-layer displacement is projected
onto the signed final-layer decision normal. Inward speed is multiplied by the current
mass inside a training-fixed margin band to produce signed boundary hazard.

Controls are:

1. unsigned change in hidden-space distance from training;
2. inward normal speed without boundary crowding; and
3. 128 coordinate-permuted final-layer normals per fit.

Primary outcomes are changes in classification error and Brier risk. The analysis is
paired across seeds and sources. Uncertainty is assessed by two-way source/seed cluster
bootstrap; no sample is treated as an independent experimental replicate.

## Frozen gates

- hazard/Error Spearman >= 0.50;
- hazard/Brier Spearman >= 0.50;
- error-change sign accuracy >= 0.75;
- hazard correlation exceeds unsigned-distance correlation by >= 0.10;
- at least four of six sources have sign accuracy >= 0.60;
- both directions have positive hazard/Error correlation;
- true-boundary correlation exceeds the 95th percentile of permuted boundaries;
- two-way cluster-bootstrap 95% lower bounds exceed zero for both hazard correlation
  and hazard-minus-distance advantage.

All gates must pass to license a larger formal experiment. No origin, direction, seed,
transition, risk, or control may be removed after target inspection.
