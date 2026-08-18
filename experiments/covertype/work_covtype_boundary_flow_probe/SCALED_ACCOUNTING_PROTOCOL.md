# Covertype 45 m signed boundary accounting diagnostic

The risk-blind 15 m velocity reliability gate failed. A posttarget scale audit found
that 45 m was the smallest tested scale at which hidden-path split-half cosine exceeded
0.5 and both error and Brier trend signs agreed in at least 80% of split halves.

This is therefore a posttarget diagnostic, not a confirmatory result. It retains all six
directions/origins and the full six-window target horizon. Every transition compares
window t with t+3 (45 m), yielding three transitions per source.

The primary predictor is class-conditioned boundary hazard: mass within a
training-fixed margin band multiplied by contemporaneous hidden displacement projected
onto the frozen model's signed final-layer normal. Controls are signed normal speed
without crowding, unsigned training-distance change, and 128 coordinate-permuted final
layer normals.

Diagnostic gates are: sign accuracy >=0.80, hazard/error Spearman >=0.50, hazard exceeds
unsigned distance by >=0.20, at least 75% of non-increasing transitions have non-positive
hazard, and the true-boundary correlation exceeds the placebo 95th percentile.
