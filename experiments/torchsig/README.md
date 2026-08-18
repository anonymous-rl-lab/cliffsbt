# Generalization Cliff — TorchSig operational evidence

This repository contains the complete TorchSig experiment ladder used to test
whether a locally estimated deployment-risk surface can be connected to an
outcome-blind mechanism-information audit.

The primary result is `formal_quadratic_v3`: a quadratic local risk surface and
an asymmetrically optimized least-observable mechanism pair. All seven
pre-target gates passed. The final run passed 12 of 13 gates; only the frozen
two-sided 95% relative-risk interval gate abstained by 0.000292 on the
lower-risk side.

Round 6 adds a constructive measurement result. A preregistered attempt to
show that the Cliff-directed subset objective outperforms `trace(Q)` at budget
27 was rejected. The common 25-channel subset was then frozen and independently
confirmed against the full 54-channel observation: it retained 84.3%–89.1% of
worst-pair information and passed all sealed target-curve and accuracy-loss
gates.

Round 7 tests whether the fixed 25-channel telemetry can warn of a future
relative-risk boundary crossing. The final independent Q90-buffered replay
passed 9/10 gates: timely warning 64.4%, safe-trajectory false alarm 11.1%, and
median lead three windows. The sole abstention was premature warning, 21.7%
against a frozen maximum of 20%.

Round 8 asks where that prospective signal comes from. A zero-training probe
and an independent fresh-seed formal replay preserve the complete pre-crossing
state multiset and terminal state but destroy temporal order. The formal
knockout passed all 11 gates over 120 matched replicates and 28,800 shuffled
replays. Ordered histories exceeded shuffled histories by 0.0301 in integrated
forecast score (95% paired-bootstrap interval [0.0275, 0.0328]); the sudden-
jump proxy warned in only 0.83% of replicates. Within the frozen local-drift
regime, persistent accumulation is therefore identified as the source of the
prospective increment beyond current state.

Round 9 removes the named TorchSig mechanism coordinate from the deployed
warning model. In a fresh-seed formal replay, the 25D blind chart reaches 71.7%
timely warning with zero stationary false alarms and median lead 2.5 windows;
both directions replicate. However, a matched 25D mean-and-variance control
reaches 72.5%. The paired training-reference gain is -0.83 percentage points
(95% bootstrap interval [-2.5, 0]). Thus blind-u warning is supported, while a
fixed training set provides no demonstrated increment beyond matched telemetry
moments. See `ROUND9_TORCHSIG_BLIND_U.md`.

Round 10 moves the training-data question upstream. Four equal-size training
distributions are retrained under paired learner seeds and evaluated on one
shared calibration stream and one shared deployment stream. In the five-seed
formal replay, every baseline path crosses the common Round 9 boundary and no
five-percent Cliff-aware path crosses it. Cliff-aware training lowers terminal
risk by 0.3331 versus baseline (training-seed-clustered 95% interval
[-0.3683, -0.29045]) and by 0.0120 versus equal-budget random broad coverage
([-0.02625, -0.00350]). It contracts `||b||` to 0.50% and `||H||F` to 3.81% of
baseline. All 8 pretarget and 16 formal gates pass. See
`ROUND10_TORCHSIG_TRAINING_INTERVENTION.md`.

Round 11 adds paired latent-signal identity. Its synchronized-crossing smoke
stops at 6/7 gates: first passage is broad in time and true position-velocity
matching does not beat the class-conditional velocity shuffle. A fresh
three-seed redesign then supports persistent distributed net boundary flux
instead of a required synchronized pulse. Round 11C removes that probe's
approximate signal-kernel confound by executing the exact tagged TorchSig 2.1.1
algorithms used by the frozen configuration. Across five fresh training seeds
and 512 paired signals per path, all 12 gates pass: median incident persistence
is 0.9525, crossing entropy is 0.8789, and Cliff-aware training reduces endpoint
risk and incident crossing by 0.27285 and 0.12148 with positive seed-cluster
intervals. This is the manuscript-level paired mechanism confirmation, but it
uses a narrow official-source NumPy runtime rather than installed PyTorch and
TorchSig packages; package-runtime replay remains required.

Post-paper Round 12 smoke experiments ask whether the mechanism can directly
rank training examples without the hidden control coordinate. Persistent
crossing, queried signed flux, and one-at-a-time counterfactual influence all
produce large absolute repair, but none beats equal-budget random deployment
coverage on a fresh terminal-risk endpoint. The series stops before a
multi-seed pilot. See `ROUND12_CONTROL_FREE_REPAIR_SMOKE_LEDGER.md`. These
negative exploratory results are not part of manuscript v5.

## Scope of the evidence

The two numerical cutoffs in the final run are

```text
tau - gamma = 0.379018
tau + gamma = 0.459018
```

They are protocol-relative comparison cutoffs, not externally grounded safety
limits. Here, `tau = 0.419018` is the fitted risk-surface intercept at the
chosen calibration center and `gamma = 0.04` is a frozen four-percentage-point
effect size. TorchSig does not provide an operational harm threshold that would
justify calling either cutoff absolutely safe or catastrophic.

Consequently this repository supports a **lower-risk versus higher-risk
separation claim**, not an absolute deployment-safety certificate. Raw result
files retain the original `safe_cliff` field names for byte-level provenance;
their corrected interpretation is documented in `CLAIMS_AND_LIMITS.md`.

## Experiment ladder

| Run | Role | Outcome |
| --- | --- | --- |
| `global_probe` | Wide-radius linear probe | Rejected: global risk surface is not sufficiently linear |
| `local_probe` | Local-region identification | Useful calibration probe; not a formal result |
| `formal_identified` | Linear frozen replay, seed 20260814 | 7/8 gates; midpoint drift prevents relative boundary crossing |
| `formal_identified_v2` | Independent linear replay, seed 20260815 | 7/8 gates; midpoint drift replicates |
| `formal_quadratic_v3` | Quadratic asymmetric replay, seed 20260816 | 7/7 pre-target gates; 12/13 final gates; relative CI abstains |
| `formal_measurement_design_v1` | 54-channel subset design, seed 20260818 | Cliff-over-trace superiority rejected at budget 27; target remains sealed |
| `formal_measurement_compression_v1` | Frozen 25D versus 54D replay, seed 20260819 | All pretarget, curve, and accuracy-loss gates pass |
| `formal_early_warning_v1` | Balanced-center warning gate, seed 20260823 | Aborted before target: phase trajectory infeasible |
| `formal_early_warning_v2_phase_heavy` | First phase-heavy warning replay, seed 20260823 | Crossing-time bias; premature-warning gate fails strongly |
| `formal_early_warning_v3_q90` | Independent Q90-buffered warning replay, seed 20260824 | 9/10 gates; premature-warning gate misses by 0.0167 |
| `precursor_order_knockout_probe` | Zero-training matched-order probe on Round 7 | 9/9 effect gates pass; requires fresh-seed confirmation |
| `formal_precursor_source_v1` | Fresh-seed mechanism source, seed 20260826 | Pretarget abort: phase direction infeasible; no target generated |
| `formal_precursor_source_v2_robust_directions` | Two-direction source, seed 20260828 | Invalidated for formal use after same-seed logging rerun changed sealed hashes |
| `formal_precursor_source_v3_deterministic` | Deterministic fresh-seed source, seed 20260830 | All 9 pretarget and 10 warning gates pass |
| `formal_precursor_mechanism_knockout_v3` | Independent fixed-terminal order knockout, seed 20260831 | All 11 mechanism and integrity gates pass |
| `round9_blind_u_probe` | First blind-u probe | Invalidated for incremental training-reference use: unfair mean-only comparator and mixed exploratory outputs |
| `round9_blind_u_probe_v2_matched` | Matched-moment blind-u probe, seeds 20260901/02 | Target probe supports formal replay; no calibration gain over matched moments |
| `formal_round9_blind_u_v1` | Fresh-seed blind-u and training-reference test, seeds 20260903/04 | Blind-u gates pass; training-reference gain fails; 14/16 overall |
| `round10_training_intervention_probe` | Two-seed, 20% training-distribution probe | Training effect is strong; random and targeted enrichment both hit the risk floor |
| `round10_training_intervention_probe_v2_5pct` | Five-percent specificity repair | Targeted advantage survives; fresh-seed formal replay justified |
| `formal_round10_training_intervention_v1` | Five fresh paired training seeds and one common deployment stream | 8/8 pretarget and 16/16 formal gates pass; Cliff-aware prevents every crossing |
| `round11_paired_sync_smoke` | Same-signal first-crossing and placebo smoke | 6/7; synchronization/velocity-matching gate fails; retained STOP |
| `round11b_distributed_flux_probe` | Three-fresh-seed replacement probe | 9/9 compatibility gates pass; standard TorchSig confirmation required |
| `round11c_official_source_flux` | Five-seed, 512-signal-per-path official-source algorithm confirmation | 12/12 gates pass; package-runtime replay remains required |
| `round12a_blind_flux_repair_smoke` | Unlabeled persistent-switch repair | 12/13; random coverage is better; retained STOP |
| `round12a_blind_flux_repair_smoke_v2_stratified` | Predicted-class-stratified blind repair | 11/14; direction ambiguity remains; retained STOP |
| `round12a_queried_flux_repair_smoke_v3` | Label-query signed-flux feasibility | Pretarget acquisition abort; no evaluation panel generated |
| `round12a_common_query_repair_smoke_v4` | Same queried pool, signed-flux ranking | 13/16; random and uncertainty are better; retained STOP |
| `round12b_influence_repair_smoke` | One-example counterfactual repair influence | 13/16; no set-level advantage; retained STOP |
| `round12c_coverage_vs_hazard_pilot` | Five-seed coverage-versus-hazard factorial pilot | 14/14; coverage dominates hazard concentration under the frozen 5% budget |

See `EXPERIMENT_LEDGER.md` for the numerical audit, and the Round 4 and Round 5
reports for the full scientific interpretation.

## Repository layout

```text
configs/       Frozen run configurations
src/           Experiment, summarization, and manifest code
tests/         Determinism, parameterization, optimizer, and result tests
results/       Raw CSV/JSON outputs and per-run hash manifests
figures/       Generated audit figures
*.md           Scientific reports, claims, and experiment ledger
SHA256SUMS.txt Root integrity manifest
```

## Environment

The original frozen runs used TorchSig 2.1.1 with CPU PyTorch 2.11.0. Rounds
8--10 executed the same TorchSig 2.1.1 NumPy signal-generation path through a
disclosed compatibility import runtime because the full tensor runtime was
unavailable in the container. Round 11C instead executes the exact tagged
TorchSig 2.1.1 algorithms exercised by its configuration through a narrow
NumPy runtime; this removes the approximate signal-kernel confound but is still
not an installed-package replay. Exact standard-runtime versions are in
`requirements.txt`, and recorded platform metadata are in `results/*`.

## Reproduce

Create an isolated Python environment and install the frozen dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the experiment ladder:

```bash
python src/run_pilot.py --config configs/pilot.json
python src/run_pilot.py --config configs/local_probe.json
python src/run_pilot.py --config configs/formal_identified.json
python src/run_pilot.py --config configs/formal_identified_v2.json
python src/run_pilot.py --config configs/formal_quadratic_v3.json
python src/summarize_round4.py
python src/summarize_quadratic_v3.py
python src/probe_measurement_design.py
python src/probe_measurement_design_centers.py
python src/run_measurement_design_formal.py
python src/run_measurement_compression_formal.py
python src/probe_early_warning_geometry.py
python src/probe_cliff_early_warning.py
python src/probe_cliff_early_warning_v2.py
python src/probe_early_warning_batch16.py
python src/run_early_warning_formal.py --config configs/formal_early_warning_v3_q90.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python src/check_determinism_smoke.py --config configs/formal_precursor_source_v3_deterministic.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python src/run_early_warning_formal.py --config configs/formal_precursor_source_v3_deterministic.json
python src/probe_precursor_order_knockout.py --config configs/formal_precursor_mechanism_knockout_v3.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python src/run_round9_blind_u.py --config configs/formal_round9_blind_u_v1.json
python src/run_round10_training_intervention.py \
  --config configs/formal_round10_training_intervention_v1.json --calibration-only
python src/run_round10_training_intervention.py \
  --config configs/formal_round10_training_intervention_v1.json
python src/run_round11_paired_sync.py \
  --config configs/round11_paired_sync_smoke.json --freeze-only
python src/run_round11b_distributed_flux.py \
  --config configs/round11b_distributed_flux_probe.json --freeze-only
python src/run_round11c_official_source_flux.py \
  --config configs/round11c_official_source_flux.json --freeze-only
python src/run_round12c_coverage_vs_hazard_pilot.py \
  --config configs/round12c_coverage_vs_hazard_pilot.json --freeze-only
# Copy the printed digest into --expected-pretarget-sha for the single allowed reveal.
python -m unittest discover -s tests -v
python src/build_manifests.py
```

`run_pilot.py` evaluates all pre-target gates before target outcome generation.
For the quadratic run, failure of any required pre-target gate aborts before
the target reveal stage.

## Primary files

- `ROUND4_TORCHSIG_OPERATIONAL_AUDIT.md`: linear replication and curvature failure.
- `ROUND5_QUADRATIC_ASYMMETRIC_PROBE.md`: quadratic repair and remaining interval abstention.
- `ROUND6_TORCHSIG_MEASUREMENT_DESIGN.md`: rejected objective-superiority claim and confirmed 25D compression result.
- `ROUND7_TORCHSIG_EARLY_WARNING.md`: sequential warning ladder and final 9/10 Q90-buffered replay.
- `ROUND8_TORCHSIG_PRECURSOR_MECHANISM.md`: fixed-terminal temporal-order knockout and fresh-seed mechanism confirmation.
- `ROUND9_TORCHSIG_BLIND_U.md`: physical-coordinate-blind warning and the rejected fixed-training-reference increment.
- `ROUND10_TORCHSIG_TRAINING_INTERVENTION.md`: controlled retraining evidence for how training support reshapes risk, observation geometry, and cliff time.
- `ROUND11_TORCHSIG_PAIRED_BOUNDARY_FLUX.md`: paired-sample synchronization failure and fresh distributed-flux replacement probe.
- `ROUND11C_TORCHSIG_OFFICIAL_SOURCE_FLUX.md`: five-seed official-source signal-kernel confirmation and clustered intervention intervals.
- `ROUND12_CONTROL_FREE_REPAIR_SMOKE_LEDGER.md`: post-paper control-free repair STOPs and the coverage-versus-ranking conclusion.
- `ROUND12C_COVERAGE_VS_HAZARD_PILOT.md`: fresh five-seed factorial evidence separating deployment coverage from hazard hit rate.
- `CLAIMS_AND_LIMITS.md`: authoritative terminology and claim boundary.
- `results/formal_quadratic_v3/quadratic_v3_ledger.json`: machine-readable final ledger.
- `figures/formal_quadratic_v3/quadratic_v3_summary.png`: final visual summary.
- `results/formal_measurement_compression_v1/checks.json`: machine-readable measurement-compression decision.
- `figures/formal_measurement_compression_v1/formal_measurement_compression_v1.png`: 25D versus 54D target curves.
- `results/formal_early_warning_v3_q90/checks.json`: machine-readable final warning decision.
- `figures/formal_early_warning_v3_q90/formal_early_warning_v3_q90.png`: final revealed-risk and forecast trajectories.
- `results/formal_precursor_mechanism_knockout_v3/checks.json`: machine-readable persistent-accumulation mechanism decision.
- `figures/formal_precursor_mechanism_knockout_v3/precursor_order_knockout_probe.png`: matched-order mechanism summary.
- `results/formal_round9_blind_u_v1/checks.json`: machine-readable blind-u and training-reference decision.
- `figures/formal_round9_blind_u_v1/formal_round9_blind_u_v1.png`: blind-u calibration, warning, and reference-swap summary.
- `results/formal_round10_training_intervention_v1/checks.json`: machine-readable training-intervention decision.
- `figures/formal_round10_training_intervention_v1/formal_round10_training_intervention_v1.png`: common-stream risk and geometry summary.
