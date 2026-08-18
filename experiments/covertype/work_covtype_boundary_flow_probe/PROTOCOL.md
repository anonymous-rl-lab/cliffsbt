# Covertype distribution-to-boundary flow smoke v1

## Question

Does the late plateau/reversal in Covertype arise because the deployment distribution
changes direction relative to the fixed model boundary, rather than because distance
from the training distribution ceases to grow?

This is a target-aware mechanism smoke test on a new fixed panel. It is not a formal
paper result. Its purpose is to decide whether a larger confirmation is scientifically
licensed.

## Frozen design

- Binary Cover Type 1 versus 2, balanced within every 15 m elevation window.
- Physical windows 2700--3330 m (42 windows). These avoid the two edge windows whose
  minority-class leftovers are too small after excluding the original source-smoke
  panel.
- For each window and class, 48 examples form the fitting pool and a disjoint 48 form
  the flow/evaluation panel. All indices from the original 46-window source-smoke panel
  are excluded before selection.
- Ascending and descending paths are evaluated from origin positions 10, 22, and 34.
- One neural-network seed is used in this <=10 minute smoke. A passing smoke licenses a
  five-seed confirmation; it is not itself confirmatory evidence.

## Model and model-relative path

A two-hidden-layer ReLU MLP is fitted only on fitting-pool examples before each origin.
The scaler, network, and decision boundary are then frozen.

For each deployment window, the class-conditional distribution is represented in:

1. the final hidden representation of the frozen MLP; and
2. standardized raw features with elevation removed.

The training-relative position is the concatenation of the two class-conditional mean
shifts. Consecutive differences define local velocity; consecutive velocity cosine
defines turning. Unsigned training distance is retained as the negative control.

## Boundary flow

For a sample with true class y, signed margin is positive when correctly classified and
negative when misclassified. Consecutive balanced windows are coupled separately within
each class by Hungarian matching. Matching is performed independently in hidden and
raw-no-elevation space.

At time t, the observed margin velocity from t-1 to t is extrapolated one step. The
resulting predicted change in error is the signed boundary flux: predicted entries into
the error region minus predicted exits. It uses no sample, risk, or margin from t+1.

Boundary crowding uses an epsilon fixed from the positive training-margin distribution.
The mechanistic hazard is crowding times inward margin velocity. The primary comparison
is signed boundary flow versus the change in unsigned distance from training.

## Frozen smoke gate

Across all eligible one-step forecasts, require:

- sign accuracy >= 0.60;
- Spearman correlation between signed flow forecast and next risk change >= 0.35;
- this correlation exceeds the unsigned-distance control by >= 0.10;
- at least 60% of late non-increasing transitions have non-positive predicted flow;
- conclusions agree in sign between hidden and raw-no-elevation coupling.

Failure stops this route or triggers a redesign supported only by diagnostic probes.
No origin, direction, window, matching space, or threshold may be narrowed after target
inspection.

## Scope limitation

The windows contain different samples. Therefore Hungarian matching estimates a
distributional transport coupling; it does not create true longitudinal identities.
Even a pass would establish a plausible probability-flow explanation, not literal
sample-level trajectories.
