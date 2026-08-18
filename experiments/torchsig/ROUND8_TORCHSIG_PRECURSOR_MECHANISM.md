# Round 8 — TorchSig precursor mechanism knockout

## Decision

**The first precursor-mechanism gate passes within the frozen local-drift
regime.** In an independent fresh-seed TorchSig replay, preserving the exact
pre-crossing telemetry-state multiset and the terminal pre-crossing state while
permuting only temporal order removed a positive, replicated component of the
five-window prospective score. All 11 formal effect, replication, negative-
control, bootstrap, and design-integrity checks passed.

This result identifies persistent temporal accumulation as the source of the
**prospective increment** supplied by the six-window trend term. It does not
claim that every physical failure has a gradual precursor, that arbitrary
future jumps are predictable, or that the protocol-relative risk boundary is
an absolute safety threshold.

## The question made falsifiable

Let \(z_t\) be the current risk coordinate recovered from the fixed 25-channel
telemetry. The frozen warning score is

\[
F_t = z_t + L[\hat v_t]_+,
\]

where \(\hat v_t\) is the ordinary least-squares slope of the last \(H=6\)
coordinates, \([x]_+=\max(x,0)\), and \(L=5\) is the forecast horizon.

For every replicate with an independently revealed crossing time \(c\), the
formal knockout uses exactly the history needed to evaluate the five decisions
\(t=c-L,\ldots,c-1\). It compares:

- **ordered:** telemetry in its original persistent-degradation order;
- **shuffled:** the identical telemetry multiset, with \(z_{c-1}\) also held
  fixed, but all earlier coordinates randomly permuted;
- **sudden-jump proxy:** stationary-safe prehistory paired with the event's
  crossing time, representing a future discontinuity with no accumulating
  prehistory.

The revealed future and crossing time are shared within each matched
comparison. Therefore current level, state inventory, sample count, terminal
state, warning equation, boundary, and future label cannot explain an
ordered-minus-shuffled difference. The intervention changes temporal
accumulation only.

The primary paired estimands are

\[
\Delta_v = \mathbb E(\bar v_{\mathrm{ordered}}-
\bar v_{\mathrm{shuffled}}), \qquad
\Delta_F = \mathbb E(\bar F_{\mathrm{ordered}}-
\bar F_{\mathrm{shuffled}}),
\]

where bars average the five pre-crossing decision windows. The null mechanism
claim is temporal exchangeability conditional on the multiset and terminal
state, which predicts no positive paired effect.

## Frozen formal source

The confirmation source is
`formal_precursor_source_v3_deterministic`, master seed 20260830. The run used
the previously selected 25-dimensional observation groups, batch size 16, 60
replicates per trajectory, two independently feasible drift directions, six
history windows, and a five-window horizon. The alarm threshold was fixed as
the fitted relative boundary plus the calibration Q90 residual buffer.

Before any target trajectory was generated, all nine identification and
geometry gates passed:

| Pretarget quantity | Result |
| --- | ---: |
| Calibration risk range | 0.390625 |
| Quadratic risk-surface R2 | 0.966058 |
| Five-fold CV R2 | 0.957542 |
| Relevant-score linear R2 | 0.994664 |
| 25D risk-null ratio | 0.039113 |
| 54D risk-null ratio | 0.032331 |
| Feasible frozen directions | 2/2 |

The relative warning boundary was 0.235669, the calibration-only Q90 buffer
was 0.030559, and the final alarm threshold was 0.266228. These are local,
protocol-relative values for this fresh-seed warning source; they are not the
Round 5 absolute-looking legacy cutoffs and not engineering safety limits.

The sealed warning replay then passed all 10 warning gates:

| Source metric | 25D forecast | Current-state baseline |
| --- | ---: | ---: |
| Timely warning | 0.6250 | 0.0583 |
| Forecast gain | 0.5667 | — |
| Premature warning | 0.1250 | 0.0000 |
| Stationary-safe false alarm | 0.0417 | 0.0000 |
| Median timely lead | 3 windows | 1 window |
| Timely warning, full 54D | 0.6417 | 0.0417 |

This source result establishes that there is prospective signal to explain.
It is not itself the mechanism test.

## Formal matched-order result

The knockout uses 120 matched event replicates and 240 fixed-terminal
permutations per replicate, for 28,800 shuffled replays. Analysis seed 20260831
was frozen separately from the source seed.

| Frozen check | Requirement | Result | Status |
| --- | ---: | ---: | --- |
| Exact state-multiset preservation | 0 violations | 0 | Pass |
| Exact terminal-state preservation | 0 violations | 0 | Pass |
| Mean integrated slope difference | >= 0.003 | 0.005398 | Pass |
| Slope 95% paired-bootstrap lower bound | > 0 | 0.004982 | Pass |
| Mean integrated forecast difference | >= 0.020 | 0.030075 | Pass |
| Forecast 95% paired-bootstrap lower bound | > 0 | 0.027460 | Pass |
| Pairwise ordered superiority | >= 0.90 | 0.9833 | Pass |
| Positive-slope fraction difference | >= 0.20 | 0.3703 | Pass |
| Any-window alarm-rate difference | >= 0.15 | 0.2672 | Pass |
| Sudden-jump proxy pre-warning | <= 0.10 | 0.0083 | Pass |
| Minimum direction forecast difference | > 0 | 0.027459 | Pass |

The complete 95% paired-bootstrap intervals were [0.004982, 0.005815] for
integrated slope and [0.027460, 0.032763] for integrated forecast score.

### Direction replication

| Direction | Ordered slope | Shuffled slope | Forecast difference | Ordered alarm | Shuffled alarm | Sudden proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed gradient | 0.005830 | 0.000886 | 0.027459 | 0.6667 | 0.3888 | 0.0000 |
| Noise axis | 0.006814 | 0.000963 | 0.032691 | 0.7833 | 0.5267 | 0.0167 |

The shuffled alarm rate does not fall to zero because the knockout deliberately
retains the same elevated terminal state. That is a feature of the matched
design: it prevents a trivial current-level explanation. The primary evidence
is the paired continuous slope/forecast loss, its positive bootstrap interval,
and replication in both directions. The 26.72-percentage-point alarm reduction
is a downstream consequence.

The 72.5% ordered any-window alarm rate in this table is not the same estimand
as the source run's 62.5% timely-warning rate. The knockout asks whether any
alarm occurs in the designated five-window interval; the source ledger assigns
each replicate by its earliest alarm, so an earlier alarm is classified as
premature rather than timely.

## Failure audit and reproducibility repair

No discarded run is promoted as evidence.

1. `formal_precursor_source_v1` (seed 20260826) aborted before target
   generation because the phase direction could not meet the frozen endpoint
   margin. Its strong identification metrics and geometry failure are retained.
2. `formal_precursor_source_v2_robust_directions` (seed 20260828) passed its
   warning gates, but the original output omitted early risk coordinates needed
   for the knockout. A same-seed logging rerun changed the hashes of core sealed
   outputs. It was therefore invalidated as a formal confirmation source. The
   before/after hashes are frozen in
   `results/formal_precursor_source_v2_robust_directions/REPRODUCTION_INVALIDATION.json`.
3. The repair made ExtraTrees `n_jobs` configurable, froze it to one, froze
   BLAS/OpenMP thread counts, and required a two-process determinism smoke test.
   The two smoke hashes matched exactly before fresh seed 20260830 was run once.
4. All-time risk-coordinate logging was present before v3 target generation.
   The sealed source hashes were recorded before the knockout, and the knockout
   reads `sealed_all_time_risk_coordinates.csv` directly.

The execution environment imported TorchSig 2.1.1 signal-generation source
through a NumPy-only compatibility runtime because a full tensor runtime was
not available in the execution container. No tensor operation is used by the
selected TorchSig generation path. `environment.json` records
`torch = compat-no-tensor-runtime`; exact package pins for a standard full
runtime remain in `requirements.txt`. This compatibility detail is a disclosed
reproduction limitation, not concealed as a standard PyTorch execution.

## Claim boundary

The strongest supported statement is:

> Conditional on the frozen TorchSig model, local calibration, 25-channel
> telemetry map, six-window linear trend rule, two feasible persistent-drift
> directions, and five-window horizon, temporal accumulation is the identified
> source of the prospective forecast increment beyond the current risk level.

The experiment does not establish universality across benchmarks or hardware,
does not distinguish every possible physical cause of accumulation, and does
not warn before an unanticipated discontinuity. The next strategic gate is an
external-system replication with naturally occurring or independently
specified degradation dynamics.

## Evidence map

- Frozen source config: `configs/formal_precursor_source_v3_deterministic.json`
- Frozen knockout config: `configs/formal_precursor_mechanism_knockout_v3.json`
- Determinism audit: `src/check_determinism_smoke.py`
- Source generator: `src/run_early_warning_formal.py`
- Matched-order analysis: `src/probe_precursor_order_knockout.py`
- Sealed source outputs: `results/formal_precursor_source_v3_deterministic/`
- Raw matched pairs and all shuffle replays:
  `results/formal_precursor_mechanism_knockout_v3/`
- Summary figure:
  `figures/formal_precursor_mechanism_knockout_v3/precursor_order_knockout_probe.png`

