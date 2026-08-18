# Fresh-holdout confirmation of robust margin transport v1

## Frozen hypothesis

The rejected global statistic used the mean hidden displacement projected onto the
model boundary. Posttarget diagnostics showed that a few extreme margin velocities
destabilized that mean, whereas the equal-quantile median margin velocity was stable
across five formal model seeds.

The frozen statistic for this confirmation is:

`current boundary density * negative median equal-quantile margin velocity`,

averaged across the two true classes. It contains no shoulder selection and no
synchronization multiplier. Those diagnostic constructions were rejected and are not
retained.

## Last known-unused target panel

- All indices in the original source smoke, first flow panel, large accounting panel,
  45 m pilot, and 45 m formal panel are excluded.
- Sixty-four examples per class are selected independently in every one of the thirteen
  45 m blocks over 2745--3330 m.
- The scarcest remaining block/class pool contains 88 examples before this selection.
- The formal fit panel is reused, but five entirely new model seeds are trained. The
  target examples and model seeds are both unseen by the robust-margin development.

## Controls

The frozen comparisons are mean margin velocity, 10% trimmed mean velocity, unsigned
hidden distance from training, and 128 coordinate-permuted model boundaries.

## Gates

All must pass:

1. median-transport/Error Spearman >=0.65;
2. error-change sign accuracy >=0.75;
3. each direction Spearman >=0.50;
4. advantage over mean transport >=0.20;
5. advantage over unsigned distance >=0.20;
6. at least four of six sources have sign accuracy >=0.60;
7. true boundary exceeds the placebo 95th percentile;
8. two-way source/seed bootstrap median-correlation lower bound >=0.40;
9. two-way bootstrap lower bounds for both advantages exceed zero.

No source, seed, direction, block, risk, or control may be removed after target release.
The 45 m scale remains conditional on the earlier posttarget scale diagnostic.
