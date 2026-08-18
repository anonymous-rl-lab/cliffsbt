# Complete experiment protocol

## 1. Question and serial claim

The experiment asks whether one frozen boundary-transport framework can close
three linked modules on paired CURE-OR deployment streams:

1. identify formation of persistent risk cliffs;
2. warn before cliff onset from fixed outcome-blind telemetry;
3. reduce risk and remove or delay cliffs with a fixed calibration-gated
   training update.

The full claim is conjunctive:

`H_full = H_formation AND H_warning AND H_repair`.

Each module is reported separately. No downstream module rescues a failed
upstream gate, and no threshold or endpoint is changed after outcome reveal.

## 2. Dataset and experimental axes

- Dataset: mini CURE-OR, DOI `10.5281/zenodo.4299330`, CC-BY-4.0.
- Classifier-head seeds: `113, 127, 139, 151, 163`.
- Schedule IDs: `211, 223, 227`.
- Target challenge families: `2, 6, 11, 12, 13, 14, 15, 16, 17, 18`.
- Calibration identities: 50 class-balanced base identities.
- Confirmation identities: 50 disjoint class-balanced base identities.
- Windows per path: 13.
- Total confirmation paths: `5 × 3 × 10 = 150`.

Schedules change only the frozen asynchronous order in which paired identities
advance through severity levels. They are repeated stress conditions, not
independent inferential units. Inference clusters by model seed.

## 3. Representation, heads, and streams

The backbone is ImageNet ConvNeXt-Tiny with expected weight SHA-256:

`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.

The backbone is frozen. For each registered seed, deterministic seed-specific
crop/flip augmentation is applied to the frozen training pool, and a ridge
classification head is fitted to 150 clean training identities. Confirmation
images are not used to fit a head.

Color families 2 and 6 start at no-challenge baseline. Grayscale families
11–18 start at grayscale baseline. Window 0 is baseline; window 12 places all
50 identities at severity 4. Intermediate windows use the frozen schedule hash.
The same base identity is followed across all windows and comparisons.

Risk is the fraction of misclassified identities in a 50-identity window. The
registered operational threshold is `tau = 0.50`. A persistent cliff begins at
the earliest window whose risk is at least 0.50 and whose every later window
also remains at least 0.50. Paths with baseline risk at or above 0.50 remain in
the raw tables but are ineligible for cliff-onset and warning denominators.

## 4. H1 — formation and exact boundary-flow closure

For paired error indicator `E_i(t)`, let `F_t` count correct-to-incorrect
transitions and `G_t` count incorrect-to-correct transitions. The exact identity
is:

`R_(t+1) - R_t = (F_t - G_t) / 50`.

H1 requires:

- at least 30 eligible persistent-cliff paths;
- at least 30 eligible non-cliff controls;
- at least one cliff for every model seed;
- maximum absolute closure error at most `1e-12`;
- positive median endpoint-minus-baseline risk on cliff paths;
- passing integrity, identity, pairing, and phase-order checks.

The event counts qualify the field; the 150 paths are not treated as independent
replicates.

## 5. H2 — Hybrid25 warning

Hybrid25 records 25 unlabeled batch telemetry channels per window. Eleven active
channels encode departure mass, selected margin and margin-change quantiles,
net/persistent departures, near-boundary occupancy, and mean representation
norm. Fourteen additional channels are committed diagnostics.

The frozen readout uses the active 11-vector, its one-window difference, its
two-window slope, and normalized time, totaling 34 readout inputs. The scaler,
coefficients, intercept, and threshold are in
`config/hybrid25_warning_model_frozen.json`. The alarm threshold is 0.87,
persistence is one window, and the earliest possible alarm is window 1.

An alarm is timely only when its first trigger strictly precedes the persistent
cliff. An alarm on an eligible non-cliff path is false. A post-cliff alarm is
not timely.

H2 requires:

- pooled timely-warning rate at least 75%;
- pooled false-alarm rate at most 20%;
- median timely lead at least two windows;
- at least four seeds with timely rate at least 60%;
- at least four seeds with false-alarm rate at most 30%;
- all alarms, scores, and predictions committed before confirmation scoring.

## 6. H3 — guarded full-coverage repair

The repair pool contains 100 family-by-class fragments. The fixed budget selects
five frozen examples per fragment, totaling 500. The update is a
baseline-anchored delta-ridge head with trust penalty `lambda = 1000`.

For each model seed, the candidate repair is evaluated on all calibration paths
and deployed only when all four fixed guards pass:

1. at least one unique family-level cliff disappears;
2. no unique family acquires a new cliff;
3. no calibration path has higher mean risk;
4. aggregate calibration mean-risk gain is at least 0.05.

If any guard fails, deployment retains the unchanged baseline head. The gate,
candidate predictions, and deployed predictions are committed in Phase 1.

H3 requires:

- at least three eligible model repairs;
- at least two unique model–family cliffs removed;
- zero newly introduced model–family cliffs;
- at least two model seeds with a removed cliff;
- a model-seed cluster-bootstrap 95% interval for mean risk gain strictly above
  zero;
- positive mean event-time gain, coding no cliff as time 13.

The bootstrap uses 20,000 replicates with NumPy RNG seed `21313`. A resampled
seed carries its complete schedules, families, windows, identities, and repair
decision. Images, paths, schedules, families, and windows are not resampled as
independent observations.

## 7. Two-phase outcome separation

Phase 1 trains heads, applies calibration, computes confirmation predictions,
logs Hybrid25 telemetry and alarms, applies the repair safety gate, and commits
the output hashes without scoring confirmation labels. Its blind-commit SHA-256
is:

`4a888493531dfa797efa765adc2057e73564eacc59a9e42c86b8ef2b27c1b237`.

Phase 2 verifies the package binding and blind commitment, reads confirmation
truth, evaluates H1–H3, and writes final evidence tables and hashes. The formal
execution preserved the phase order and passed all integrity checks.

## 8. Interpretation boundary

The exact paired transition accounting is architecture-independent for a fixed
deterministic classifier and paired identities. The empirical warning and
repair results are conditional on this dataset, target calibration, frozen
telemetry/readout, population scale, pairing, schedules, and training protocol.
The experiment does not establish a universal warning sensor, universal repair
allocation rule, external safety threshold, or fully label-free end-to-end
system.
