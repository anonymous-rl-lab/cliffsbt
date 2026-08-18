# Round 9 — Blind-u warning and the role of fixed training data

## Decision

Round 9 separates two claims that were previously conflated.

1. **A named physical mechanism coordinate is not required by the deployed
   warning rule.** This claim passes in the fresh-seed formal replay. A scalar
   risk chart learned directly from calibration outcomes and outcome-blind 25D
   telemetry warns of the frozen relative-risk crossing without receiving the
   three TorchSig impairment coordinates.
2. **The realized training set adds warning information beyond matched
   telemetry moments for a fixed model.** This stronger claim fails. The
   training-relative chart is statistically indistinguishable from, and
   slightly worse than, an equally expressive 25D moment chart.

The correct scientific conclusion is therefore:

> Round 9 removes semantic knowledge of (u) from the deployed predictor, but
> it does not show that a fixed training dataset contributes incremental
> predictive information once the model and matched outcome-blind telemetry
> are available.

This result directs the next experiment upstream: vary the training
distribution, retrain the model, and test whether the same deployment stream
produces different risk geometry. Treating a fixed training set as just
another input is not supported as the main route.

## Experimental question

The prior warning pipeline estimated the three-dimensional TorchSig state
(u) from telemetry, then evaluated a quadratic risk surface. Real deployments
may not expose named impairment controls. Round 9 asks whether a risk-sufficient
scalar chart can instead be learned directly:

\[
z_t = \widehat r(\Psi(O_{t,1:B},D_{\mathrm{train}})),
\]

where (O_{t,1:B}) is the outcome-blind batch telemetry and the blind models
never receive (u). The experiment generator retains (u) only to construct
controlled trajectories and the oracle comparator.

## Frozen channels

| Channel | Dimension | Uses training reference? | Purpose |
| --- | ---: | --- | --- |
| `blind25_mean` | 25 | No | Minimal 25D telemetry mean |
| `blind25_moments` | 50 | No | Fair mean + log-variance control |
| `blind54_mean` | 54 | No | Full telemetry-mean control |
| `train25_relative` | 52 | Yes | Diagonal-whitened mean, variance ratio, and two reference-distance summaries |
| `train54_relative` | 110 | Yes | Full-channel training-relative chart |
| `train25_wrong_swap` | 52 | Deliberately wrong | Frozen-reference placebo |
| `oracle25_u` | 3-state bridge | Uses true calibration coordinates | Comparator only |

Every blind risk model is fitted on batch-16 calibration summaries and tested
by leaving complete calibration environments out. This repair was required
after the first calibration probe showed that fitting environment means and
deploying on smaller batches created an avoidable scale mismatch.

## Stepwise probe and retained failure

The first probe produced an apparently large timely-warning gain for the
training-relative chart, 0.500 versus 0.125. That contrast was invalid for an
incremental training-data claim because the comparator used only means while
the training-relative chart also used batch variance and tail information.

The matched-moment probe repaired the comparison:

| Method | Timely warning | False alarm | Position MAE |
| --- | ---: | ---: | ---: |
| `blind25_moments` | 0.3125 | 0.0625 | 0.02313 |
| `train25_relative` | 0.5000 | 0.0625 | 0.02347 |
| `oracle25_u` | 0.2500 | 0.2500 | 0.02552 |
| Wrong-reference swap | 0.0000 | 0.0000 | 0.35000 |

This eight-replicate-per-direction probe justified a fresh formal test but not
a conclusion. The original mixed-output folder is explicitly invalidated and
retained only as failure provenance.

## Formal protocol

- Calibration/master seed: `20260903`.
- Fresh target seed: `20260904`.
- Training data: 500 samples per class, four classes, 2,000 total.
- Model: 240-tree ExtraTrees, single-thread fit.
- Calibration: 125 environments, 256 labelled samples per environment.
- Deployment batch: 16 unlabelled observations.
- Trajectories: noise and mixed-gradient event paths plus matched stationary
  controls.
- Replicates: 60 per trajectory, giving 120 event and 120 stationary-control
  warning records per method.
- History: six windows; forecast horizon: five windows.
- Risk reveal: 2,000 labelled samples per unique state.
- Uncertainty: 10,000-draw paired bootstrap over matched event replicates for
  the training-reference timely-warning contrast.

All seven pretarget gates passed before target generation. Six released files
were hashed in `PRETARGET_RELEASE_SHA256.txt`; every hash remained unchanged
after the target replay.

## Pretarget result

| Quantity | Result |
| --- | ---: |
| Risk range | 0.234375 |
| Oracle quadratic CV (R^2) | 0.906950 |
| 25D oracle risk-null ratio | 0.070936 |
| `train25_relative` environment CV (R^2) | 0.898619 |
| `train25_relative` environment CV Spearman | 0.944251 |
| `train25_relative` batch-16 CV (R^2) | 0.868393 |
| `train25_relative` batch-16 MAE | 0.014523 |
| Feasible trajectory directions | 2/2 |

The matched moment control has batch-16 MAE 0.014562. The training-relative
calibration difference is negligible, already suggesting nonincrementality.

## Formal warning result

| Method | Timely warning | Premature | Stationary false alarm | Median lead | Position MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blind25_mean` | 0.7000 | 0.0083 | 0.0000 | 2.0 | 0.007200 |
| `blind25_moments` | 0.7250 | 0.0000 | 0.0000 | 2.0 | 0.007345 |
| `train25_relative` | 0.7167 | 0.0000 | 0.0000 | 2.5 | 0.007355 |
| `train54_relative` | 0.7250 | 0.0000 | 0.0000 | 2.0 | 0.007604 |
| `oracle25_u` | 0.7750 | 0.0583 | 0.0000 | 3.0 | 0.008822 |
| Wrong-reference swap | 0.0000 | 0.0000 | 0.0000 | — | 0.159031 |

The contemporaneous `train25_relative` detector has zero timely warnings, so
adding the six-window velocity contributes 0.7167 timely-warning rate. The two
formal directions replicate:

| Direction | `train25_relative` timely warning | False alarm |
| --- | ---: | ---: |
| Mixed gradient | 0.8167 | 0.0000 |
| Noise | 0.6167 | 0.0000 |

Thus the deployment predictor can recover a useful risk position and velocity
without receiving the named mechanism coordinate.

## Training-reference decision

Against the matched 25D moment control:

\[
\Delta_{\mathrm{timely}}
=0.7167-0.7250=-0.00833,
\]

with paired-bootstrap interval

\[
[-0.025,\ 0].
\]

The position-MAE improvement is

\[
0.0073453-0.0073545=-9.24\times10^{-6},
\]

with paired-bootstrap interval

\[
[-2.47\times10^{-5},\ 6.61\times10^{-6}].
\]

The training-relative chart is operationally adequate but does not improve on
matched telemetry moments. Its two designated warning-gain gates fail. The
overall formal status is 14/16 gates, with both failures belonging to the
incremental training-reference claim; every blind-u, direction, false-alarm,
lead, position, and placebo gate passes.

## Meaning of the wrong-reference result

Replacing the realized training reference after fitting sends the risk chart
to an almost constant zero coordinate. Position MAE rises by 0.15168 and timely
warning falls by 0.7167. This does not prove that the correct training set adds
information. It shows that a model built in training-relative coordinates is
sensitive to its declared origin. A wrong coordinate origin can destroy the
instrument even when a correctly calibrated reference does not outperform an
equally expressive reference-free chart.

## Claim boundary

Round 9 establishes a **control-blind deployed readout**, not a fully
mechanism-free experimental system. Calibration outcomes are still required,
and the controlled benchmark uses the hidden TorchSig coordinates to generate
and evaluate trajectory challenges. Real systems still need a way to collect
calibration environments spanning risk-relevant variation.

The stronger next experiment is therefore not another fixed-reference feature
engineering round. It is a training-distribution intervention:

\[
D_k \longrightarrow W_k \longrightarrow
(\tau_k,b_k,H_k,Q_k,t_{\mathrm{cliff},k}),
\]

with multiple models trained on support-depleted, baseline, random-enriched,
and Cliff-aware-enriched datasets and evaluated on the exact same deployment
streams.
