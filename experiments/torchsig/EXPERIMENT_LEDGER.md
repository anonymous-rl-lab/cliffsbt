# Experiment ledger

## Frozen run sequence

| Run | Seed | Surface | Main audit result | Gate status |
| --- | ---: | --- | --- | ---: |
| `global_probe` | 20260814 | Wide linear | Risk surface not globally linear; retained rejection | 6/7 |
| `local_probe` | 20260814 | Local linear | Identified useful local operating region | 7/9 |
| `formal_identified` | 20260814 | Local linear | Information curve replicates; relative point gate fails | 7/8 |
| `formal_identified_v2` | 20260815 | Local linear | Independent replication; midpoint drift persists | 7/8 |
| `formal_quadratic_v3` | 20260816 | Local quadratic | Structural repair succeeds; interval gate abstains | 12/13 |
| `formal_measurement_design_v1` | 20260818 | Local quadratic + channel subsets | Cliff-over-trace gain is 1.00 at budget 27; target not generated | Designated gain gate fails at both centers |
| `formal_measurement_compression_v1` | 20260819 | Frozen 25D versus full 54D | 84.3%–89.1% information retained; target curves confirm compression | All gates pass |
| `formal_early_warning_v1` | 20260823 | Balanced-center sequential warning | Phase direction cannot meet endpoint margin; no target generated | Pretarget geometry abort |
| `formal_early_warning_v2_phase_heavy` | 20260823 | Phase-heavy sequential warning | Warning is systematically too early | 6/10 final gates |
| `formal_early_warning_v3_q90` | 20260824 | Q90-buffered 25D warning | Timely 0.644, false alarm 0.111, median lead 3 | 9/10; premature 0.217 > 0.20 |
| `precursor_order_knockout_probe` | 20260825 | Round 7 sealed telemetry | Same multiset and terminal state; order-only effect is positive | 9/9 probe gates |
| `formal_precursor_source_v1` | 20260826 | Fresh warning source | Phase endpoint is infeasible | Pretarget abort; no target |
| `formal_precursor_source_v2_robust_directions` | 20260828 | Two feasible directions | Same-seed logging rerun changes sealed hashes | Invalidated for formal mechanism use |
| `formal_precursor_source_v3_deterministic` | 20260830 | Single-thread fresh warning source | Timely 0.625, false alarm 0.0417, median lead 3 | 9/9 pretarget; 10/10 warning |
| `formal_precursor_mechanism_knockout_v3` | 20260831 | Fixed-terminal temporal-order knockout | Forecast effect 0.0301; sudden proxy 0.0083 | 11/11 mechanism/integrity gates |
| `round9_blind_u_probe` | 20260901/02 | Blind-u plus training reference | Unfair mean-only comparator discovered; mixed exploratory folder retained | Invalidated for claims |
| `round9_blind_u_probe_v2_matched` | 20260901/02 | Matched-moment blind-u probe | Formal replay justified; training-reference calibration gain absent | Probe final gates pass; pretarget increment gate fails |
| `formal_round9_blind_u_v1` | 20260903/04 | Fresh-seed blind-u and training-reference replay | Blind-u timely 0.7167; matched moments 0.7250 | 14/16; two training-gain gates fail |
| `round10_training_intervention_probe` | 20261001/02 | Four training distributions, 20% enrichment | Both enriched arms hit near-zero risk; retained floor effect | Scale-up passes; specificity saturated |
| `round10_training_intervention_probe_v2_5pct` | 20261001/02 | Equal-budget 5% specificity repair | Targeted terminal risk is lower than random broad | Fresh-seed formal replay justified |
| `formal_round10_training_intervention_v1` | 20261011–15 | Controlled retraining on one shared deployment stream | Baseline crosses 10/10; Cliff-aware crosses 0/10 | 8/8 pretarget; 16/16 formal |

## Final quadratic run

### Pre-target gate

| Metric | Result | Status |
| --- | ---: | --- |
| Risk-surface training R2 | 0.9766 | Pass |
| Five-fold risk-surface CV R2 | 0.9710 | Pass |
| Relevant-score linear R2 | 0.9853 | Pass |
| Risk-null ratio | 0.1123 | Pass |
| Pair optimizer constraint error | 4.70e-15 | Pass |
| Minimum support slack | -1.58e-11 | Pass within tolerance |
| Five-fold pair constraint error | 0.00231 | Pass |

All seven gates passed before target outcomes were generated.

### Revealed relative-risk audit

| State | Point error | 95% Wilson interval | Frozen relative cutoff | Result |
| --- | ---: | --- | ---: | --- |
| Lower-risk state | 0.3698 | [0.360390, 0.379310] | <= 0.379018 | Point pass; interval abstain by 0.000292 |
| Higher-risk state | 0.4890 | [0.479209, 0.498800] | >= 0.459018 | Point and interval pass |

The realized midpoint drift was 0.01038, below the frozen model buffer of 0.02.
The realized half-gap was 0.0596, or 1.49 times the certification effect size.

### Information curve

| Unlabelled batch size | Quadratic theory | Frozen auditor | Target-label oracle |
| ---: | ---: | ---: | ---: |
| 2 | 0.8302 | 0.8900 | 0.8520 |
| 4 | 0.9116 | 0.9640 | 0.9520 |
| 8 | 0.9719 | 0.9940 | 0.9760 |
| 16 | 0.9965 | 1.0000 | 1.0000 |
| 32 | 0.9999 | 1.0000 | 1.0000 |
| 64 | 1.0000 | 1.0000 | 1.0000 |

Final decision: **quadratic structural repair supported; relative interval
certificate abstains; absolute deployment safety not tested.**

## Frozen measurement-compression run

The preregistered design-superiority run first rejected the claim that the
Cliff-directed objective beats `trace(Q)` at the primary budget of 27. Both
objectives selected the same 25-dimensional channel at both centers, giving a
gain of exactly 1.00 against a required 1.05. No target outcomes were generated
in that run.

The shared subset was then frozen and replayed independently with seed
20260819:

`model_outputs + complex_moments + autocorrelation + spectral`

| Center | Cost | Information retention versus 54D | 25D curve MAE | 25D Spearman | Mean full-minus-25D accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | 25/54 | 0.8432 | 0.0288 | 0.9856 | -0.00056 |
| Phase-heavy | 25/54 | 0.8911 | 0.0247 | 0.9856 | -0.00056 |

Final measurement decision: **fixed channel compression supported within the
two frozen TorchSig regions; unique Cliff-objective superiority not supported.**

## Frozen sequential-warning run

The final warning rule maps the fixed 25D telemetry to the quadratic local risk
coordinate, estimates its trend over six windows, and forecasts five windows
ahead. It alarms only when the forecast exceeds the relative boundary plus the
calibration risk-surface CV residual Q90.

| Metric | 25D forecast | 54D forecast | Frozen 25D gate |
| --- | ---: | ---: | ---: |
| Timely warning | 0.6444 | 0.6944 | >= 0.60 |
| Premature warning | 0.2167 | 0.1722 | <= 0.20 |
| Safe-trajectory false alarm | 0.1111 | 0.0889 | <= 0.25 |
| Median timely lead | 3 | 3 | >= 1 |

The 25D current-state detector reached only 0.1278 timely warning, so the
forecast gain was 0.5167. All direction-specific gates passed. The overall
premature-warning gate failed by 0.0167, equal to three event replays beyond the
allowed count.

Final warning decision: **prospective signal supported under persistent local
drift; strict warning certificate abstains at 9/10 gates.**

## Frozen precursor-mechanism run

The Round 8 formal source uses the same fixed 25-dimensional telemetry groups,
six history windows, and five-window forecast horizon. It independently passed
all source gates before the mechanism test: timely warning 0.625, false alarm
0.0417, premature warning 0.125, median lead three windows, and forecast gain
0.5667 over current-state detection.

The formal knockout holds the full pre-crossing state multiset and final
pre-crossing state exactly fixed, then permutes only earlier temporal order.

| Mechanism quantity | Frozen gate | Result |
| --- | ---: | ---: |
| State-multiset violations | = 0 | 0 |
| Terminal-state violations | = 0 | 0 |
| Mean integrated slope difference | >= 0.003 | 0.005398 |
| Slope bootstrap interval | lower > 0 | [0.004982, 0.005815] |
| Mean integrated forecast difference | >= 0.020 | 0.030075 |
| Forecast bootstrap interval | lower > 0 | [0.027460, 0.032763] |
| Ordered pair superiority | >= 0.90 | 0.9833 |
| Positive-slope fraction difference | >= 0.20 | 0.3703 |
| Any-window alarm difference | >= 0.15 | 0.2672 |
| Sudden-jump proxy pre-warning | <= 0.10 | 0.0083 |
| Minimum direction forecast effect | > 0 | 0.027459 |

Final mechanism decision: **within the frozen local persistent-drift regime,
temporal accumulation is the identified source of the prospective forecast
increment beyond current state. Universal or discontinuous-failure warning is
not established.**

## Frozen blind-u and training-reference run

The formal Round 9 readout receives no physical impairment coordinate. It maps
batch-16 25D telemetry directly to a scalar risk coordinate using labelled
calibration environments, then applies the frozen six-window, five-step trend
rule.

| Quantity | `train25_relative` | Matched `blind25_moments` | Oracle-u comparator |
| --- | ---: | ---: | ---: |
| Calibration batch CV R2 | 0.8684 | 0.8677 | — |
| Sealed position MAE | 0.007355 | 0.007345 | 0.008822 |
| Timely warning | 0.7167 | 0.7250 | 0.7750 |
| Premature warning | 0.0000 | 0.0000 | 0.0583 |
| Stationary false alarm | 0.0000 | 0.0000 | 0.0000 |
| Median lead | 2.5 | 2.0 | 3.0 |

The paired timely-warning gain from the training reference is -0.00833 with
95% bootstrap interval [-0.025, 0]. The matched position difference is also
null. A deliberately wrong reference increases position MAE from 0.00735 to
0.15903 and suppresses every warning, demonstrating coordinate sensitivity but
not positive information gain.

Final Round 9 decision: **named-u dependence of the deployed readout is removed;
the fixed-training-reference increment is rejected. The stronger training-data
question moves upstream to controlled retraining interventions.**

## Frozen training-distribution intervention

Round 10 retrains four equal-size models for each of five fresh training seeds.
Every arm uses 2,000 balanced examples and the same learner random state within
seed. One 32,000-example calibration cache and one 84,000-example deployment
cache are reused by all 20 models.

| Arm | Mean start risk | Mean end risk | Mean risk area | Crossing fraction |
| --- | ---: | ---: | ---: | ---: |
| Support depleted | 0.2226 | 0.3936 | 0.2938 | 1.00 |
| Baseline | 0.1758 | 0.3341 | 0.2401 | 1.00 |
| Random broad, 5% | 0.0086 | 0.0130 | 0.00945 | 0.00 |
| Cliff-aware, 5% | 0.0012 | 0.0010 | 0.00110 | 0.00 |

| Clustered paired contrast | Estimate | 95% interval |
| --- | ---: | ---: |
| Cliff-aware minus baseline, end risk | -0.33310 | [-0.36830, -0.29045] |
| Cliff-aware minus random broad, end risk | -0.01200 | [-0.02625, -0.00350] |
| Cliff-aware minus random broad, risk area | -0.00835 | [-0.01745, -0.00265] |

Cliff-aware training contracts the mean local gradient norm from 1.0555 to
0.00526 and the Hessian Frobenius norm from 2.4201 to 0.0923. The fixed-25D
`trace(Q)` changes from 155.785 to 134.783; this is evidence that training also
changes observation geometry, not that a smaller trace is intrinsically
better. Ten models retain active risk geometry and pass the frozen fit and
null-ratio gates; the ten enriched models are classified as weak/vanished risk
geometry rather than failed observability.

Final Round 10 decision: **training support causally shapes risk geometry and
cliff time inside the controlled TorchSig system. Five-percent targeted
coverage is more efficient than equal-budget random broad coverage on the
continuous risk endpoints, while both prevent the frozen boundary crossing.**

## Paired sample-boundary probes

Round 11A preserves the identity of each latent signal across all deployment
windows. Exact forward-minus-recovery flux reconstructs every risk increment,
both Baseline paths cross their relative boundary, both real model boundaries
beat accuracy-matched random boundaries, and Cliff-aware training reduces
endpoint risk and incident crossing. The frozen velocity-matching gate fails on
both paths, producing the retained decision `SMOKE_STOP_REDESIGN` (6/7).

A posttarget failure diagnosis finds one shuffle-q95 exceedance in 22 active
transitions (one-sided binomial p=0.6765), first-crossing time IQRs of 4.0 and
5.5 windows, and normalized entropies of 0.835 and 0.844. A synchronized pulse
is therefore not supported as necessary.

Round 11B freezes a persistent distributed-flux replacement and uses three new
model seeds plus a new 320-signal paired panel per path. All 9/9 redesign gates
pass: Baseline crossing fraction 1.0, median incident persistence 0.959, median
first-crossing entropy 0.827, largest three-window incident share 0.380, random-
boundary specificity 6/6, zero extra velocity-coupling pairs, and Cliff-aware
endpoint/incident reductions 0.2224/0.0828. Decision:
`DISTRIBUTED_FLUX_COMPAT_PROBE_PASS_STANDARD_RUNTIME_REQUIRED`.

Rounds 11A and 11B are compatibility-runtime probes. They update the next
hypothesis but do not themselves enter the manuscript-level confirmation
ladder. Round 11C below removes their approximate signal-kernel confound and is
the qualified source-faithful confirmation; installed-package replay remains a
separate runtime audit.

## Official-source boundary-flux confirmation

Round 11C executes the exact TorchSig 2.1.1 signal algorithms used by the
frozen experiment through a narrow NumPy runtime. It uses five fresh training
seeds and a fresh 512-sample paired panel on each path. All 12/12 gates pass:
Baseline cliffs occur on 10/10 model-path pairs; the maximum flux-accounting
error is zero; median incident persistence is 0.9525; median crossing entropy
is 0.8789; and the largest three-window share is 0.3827. No Baseline pair
requires extra velocity coupling, while 9/10 beat random-boundary q95.

Cliff-aware training reduces endpoint risk by 0.27285 with training-seed
cluster interval [0.21543, 0.33496], and reduces incident crossings by 0.12148
with interval [0.08574, 0.15176]. Both paths improve. Decision:
`OFFICIAL_SOURCE_NUMPY_CONFIRMATION_PASS_PACKAGE_RUNTIME_REQUIRED`.

This removes the approximate signal-kernel confound but is not execution with
the installed PyTorch/TorchSig wheel. Package-runtime confirmation remains
outstanding.

## Post-paper control-free repair exploration

Rounds 12A and 12B test the stronger prescriptive claim that the paired
boundary-transport mechanism can identify a more efficient small training set
without TorchSig controls. Four completed smokes stop and one acquisition stage
aborts before evaluation. Mechanism-ranked repair always improves substantially
over baseline, but equal-budget random deployment coverage remains as good or
better on fresh terminal risk. No stage advances to a multi-seed pilot.

Final exploratory interpretation: **boundary transport is a mechanism state,
not yet a sufficient pointwise acquisition value function.** Full numbers and
claim limits are in `ROUND12_CONTROL_FREE_REPAIR_SMOKE_LEDGER.md`.
