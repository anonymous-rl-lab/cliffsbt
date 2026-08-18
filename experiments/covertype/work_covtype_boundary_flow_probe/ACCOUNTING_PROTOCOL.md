# Covertype signed boundary accounting probe v1

## Reason for redesign

The frozen forecasting smoke found that signed hidden-space flow outperformed unsigned
training distance overall, but it failed every late non-increase case. A posttarget
split-half diagnostic then showed that distance was reliable (Spearman 0.858) while
48-per-class high-dimensional velocity was not (median split-half cosine 0.063).

This redesign addresses only that identified measurement failure. It does not narrow
directions, origins, windows, models, or outcomes.

## Larger fixed flow panel

For every one of the same 42 windows, 48 examples per class form the fit pool and 256
per class form the flow panel. The original 46-window source-smoke panel remains
excluded. The redesigned panel is diagnostic and may overlap the first failed 48-sample
flow panel; it is therefore not an independent confirmation set.

## Corrected scientific question

The first smoke conflated mechanism accounting with one-step forecasting. A path may
turn between windows, so extrapolating yesterday's velocity is not required for a valid
description of today's boundary crossing.

This probe asks whether the contemporaneous deployment velocity from window t to t+1,
projected onto the frozen model's true decision-boundary normal and multiplied by mass
near that boundary, explains the observed change in classification error.

For each class, the final hidden-layer mean displacement is projected onto the signed
final-layer weight vector. The inward normal speed is multiplied by the fraction of
samples within a training-fixed margin band around zero. Class-specific contributions
are averaged to form the signed boundary hazard.

Unsigned change in training-relative hidden distance is the principal negative
control. A second placebo permutes the coordinates of the trained final-layer normal
128 times while leaving the observed distribution path unchanged.

## Gates

Before inspecting redesigned risks, require:

- split-half velocity cosine >= 0.50;
- hazard/risk-change sign accuracy >= 0.65;
- hazard versus risk-change Spearman >= 0.50;
- hazard correlation exceeds unsigned-distance correlation by >= 0.20;
- >=60% of late non-increasing transitions have non-positive hazard;
- the true-boundary correlation exceeds the 95th percentile of permuted-boundary
  correlations.

A pass licenses a five-seed confirmation. It does not make this reused diagnostic panel
formal evidence.
