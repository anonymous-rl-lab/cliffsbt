# Round 10 — Training-distribution intervention

## Decision

Round 10 answers the upstream question left open by Round 9. The realized
training set is not useful merely as an extra fixed input to the deployed risk
chart, but the **training distribution is a causal control variable inside the
frozen TorchSig simulator**:

\[
D_k \longrightarrow W_k \longrightarrow
(\tau_k,b_k,H_k,Q_k,t_{\mathrm{cliff},k}).
\]

Changing only the training distribution, retraining the learner, and replaying
the exact same calibration and deployment samples produces large, replicated
changes in risk level, local linear and quadratic risk geometry, observation
information, and boundary-crossing time.

At the frozen five-percent enrichment budget, Cliff-aware training reduces
terminal deployment risk by 0.3331 relative to baseline, with a training-seed-
clustered 95% bootstrap interval of [-0.3683, -0.29045]. All ten baseline
seed-by-direction paths cross the shared Round 9 boundary, while none of the ten
Cliff-aware paths crosses it.

The stronger efficiency claim also passes, but with a much smaller effect.
Against equal-budget random broad coverage, Cliff-aware training lowers
terminal risk by 0.0120, interval [-0.02625, -0.00350], and lowers whole-path
risk area by 0.00835, interval [-0.01745, -0.00265]. Every one of the ten
seed-by-direction pairs favors Cliff-aware training on terminal risk.

## Why the primary boundary is shared

A model-specific boundary such as \(\tau_k+0.04\) moves whenever training
changes the model. It therefore cannot answer whether one trained model reaches
the same deployment requirement later than another. Round 10 uses the
independent Round 9 formal boundary

\[
r_{\mathrm{shared}}=0.19984587722852512
\]

for the primary cross-model comparison. Each model's \(\tau_k+0.04\) crossing
is retained only as a supplementary, moving-origin quantity. The shared
boundary remains protocol-relative and is not an external safety threshold.

## Controlled training arms

Every model receives 2,000 balanced training examples, the same 240-tree
ExtraTrees specification, and the same model random state within each paired
training seed.

| Arm | Training support | Purpose |
| --- | --- | --- |
| `support_depleted` | Uniform cube [-0.95, -0.65]^3 | Worse support-overlap control |
| `baseline` | Uniform cube [-0.8, -0.4]^3 | Round 9-style reference training |
| `random_broad` | 95% baseline + 5% uniform [-0.8, 0.8]^3 | Equal-budget generic coverage |
| `cliff_aware` | 95% baseline + 5% tube around the two frozen deployment paths | Equal-budget targeted coverage |

Five percent means 100 enriched examples per model. Class balance, total
sample count, learner hyperparameters, calibration inputs, and deployment
inputs are held fixed. The intervention therefore changes distributional
placement, not data volume or model capacity.

## Stepwise probes and retained saturation result

The first two-seed probe used a 20% enrichment dose. Both random broad and
Cliff-aware training drove deployment risk to approximately zero. Cliff-aware
terminal risk was 0–0.001, but random broad risk was already 0–0.008. This was
strong evidence that training support matters, but a poor test of targeted
efficiency because both arms hit the risk floor.

The floor also made the fitted risk direction nearly vanish. In that regime a
large risk-null ratio is not evidence that the observation channel became
blind: \(b\) itself is too small to define a stable risk direction. The
analysis was therefore repaired before formal target generation by:

1. reducing both enrichment arms to five percent;
2. freezing terminal risk and risk area as continuous specificity endpoints;
3. classifying models with calibration risk range below 0.08 or
   \(\lVert b\rVert<0.10\) as weak/vanished risk geometry rather than failed
   observability; and
4. restricting risk-surface and risk-null gates to models with an active risk
   geometry.

The five-percent two-seed probe preserved the strong training effect and
showed a nonzero targeted advantage, justifying a fresh-seed formal replay.
Neither probe is used as formal evidence.

## Formal protocol

- Training seeds: 20261011–20261015, all fresh relative to both probes.
- Evaluation seed: 20261020.
- Models: 20 total, four arms for each of five paired seeds.
- Calibration: 125 environments and 256 labelled samples per environment.
- Deployment: the frozen Round 9 noise and mixed-gradient paths, each with 21
  windows and 2,000 labelled reveal samples per window.
- Shared evaluation: one 32,000-example calibration cache and one 84,000-
  example deployment cache are generated and reused unchanged by every model.
- Crossing: two consecutive windows at or above the shared boundary.
- Uncertainty: 20,000-draw paired bootstrap clustered by the five training
  seeds; the two directions are averaged within seed for interval estimation.
- Fixed 25D channel: model outputs, complex moments, autocorrelation, and
  spectral groups.

The calibration-only pretarget run passed all eight gates before any formal
deployment sample was generated. Its eight released files have joint SHA256
`3d2188af2e47b58a757bd704f71b2499ffa5643e18cb4f65c7d4011583f7e880`.
The same joint hash was recovered after target replay.

## Formal deployment result

| Training arm | Start risk | End risk | Risk area | Shared cliff time | Crossing fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Support depleted | 0.2226 | 0.3936 | 0.2938 | 0.1 | 1.00 |
| Baseline | 0.1758 | 0.3341 | 0.2401 | 7.0 | 1.00 |
| Random broad, 5% | 0.0086 | 0.0130 | 0.00945 | censored at 21 | 0.00 |
| Cliff-aware, 5% | 0.0012 | 0.0010 | 0.00110 | censored at 21 | 0.00 |

Values average the five training seeds and two frozen directions. The support-
depleted arm increases starting risk by 0.04685 relative to baseline and is
already above the shared boundary in nearly every path. This negative control
confirms that worse training support moves the same deployment stream toward
the cliff rather than merely adding symmetric model noise.

| Frozen contrast | Estimate | Training-seed-clustered 95% interval |
| --- | ---: | ---: |
| Cliff-aware minus baseline, terminal risk | -0.33310 | [-0.36830, -0.29045] |
| Cliff-aware minus baseline, risk area | -0.23899 | [-0.27026, -0.20198] |
| Cliff-aware minus random broad, terminal risk | -0.01200 | [-0.02625, -0.00350] |
| Cliff-aware minus random broad, risk area | -0.00835 | [-0.01745, -0.00265] |

All 16 formal integrity, effect, specificity, geometry, and observation gates
pass.

## How training reshapes b, H, and Q

| Quantity | Baseline | Cliff-aware | Cliff-aware / baseline |
| --- | ---: | ---: | ---: |
| \(\tau\) | 0.219891 | 0.001220 | 0.00555 |
| \(\lVert b\rVert_2\) | 1.055517 | 0.005255 | 0.00498 |
| \(\lVert H\rVert_F\) | 2.420089 | 0.092255 | 0.03812 |
| \(\mathrm{tr}(Q)\), fixed 25D channel | 155.7854 | 134.7827 | 0.86518 |

The paired \(\mathrm{tr}(Q)\) difference is -21.0027 with interval
[-23.1758, -18.8066], so training changes the observation geometry as well as
the risk surface. This decrease is not interpreted as worse monitoring.
Cliff-aware training makes both the local risk slope and curvature nearly
vanish; when there is almost no risk change to measure, inverse risk-coordinate
information and risk-null ratios become scale-unstable. The constructive result
is the collapse of the cliff, not an alleged improvement in every scalar
summary of \(Q\).

The complete symmetric \(3\times3\) \(Q\) entries, eigenvalues, risk-null
ratios, and risk-coordinate variances for all 20 models are retained in
`model_geometry.csv`.

## Scientific meaning

Round 9 showed that a fixed realized training set should not be treated as a
magic extra feature for a fixed deployed model. Round 10 shows the stronger and
more useful role:

> Study the distribution before training, estimate where it leaves the model
> exposed, and intervene on that support. Training data shape the risk field
> that later monitoring is asked to observe.

This changes the project from diagnosis alone to a closed loop:

1. measure the deployed risk geometry and its observable directions;
2. identify unsupported, high-risk trajectories;
3. allocate new training data to those trajectories;
4. retrain; and
5. remeasure \((b,H,Q,t_{\mathrm{cliff}})\) on a common deployment challenge.

The modest but replicated advantage over random broad coverage is the first
evidence that Cliff measurement can guide data acquisition more efficiently
than generic expansion. The much larger baseline-to-enriched effect shows that
support coverage itself remains the dominant factor in this benchmark.

## Claim boundary

1. The intervention is causal only within the controlled TorchSig simulator
   and frozen ExtraTrees family. It does not establish the same magnitude for
   another model, modality, or real RF receiver.
2. Cliff-aware sampling uses the hidden TorchSig path coordinates. It does not
   solve the real-deployment problem of discovering a useful latent mechanism
   coordinate when no simulator controls are available.
3. The shared boundary is inherited from Round 9 and remains protocol-relative,
   not an engineering harm threshold.
4. The five formal training seeds support paired replication but not a broad
   population claim over training algorithms.
5. Both five-percent enrichment methods prevent crossing. Their difference is
   established only on continuous terminal-risk and risk-area endpoints, not
   on crossing fraction.
6. A vanishing \(b,H\) makes risk-directed summaries of \(Q\) ill-conditioned.
   The experiment establishes that \(Q\) changes, not that lower
   \(\mathrm{tr}(Q)\) is intrinsically desirable.

