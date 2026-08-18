# Claims and limits

This file is the authoritative semantic addendum for the TorchSig operational
experiments. It supersedes the informal use of `safe`, `cliff`, and
`certificate` in legacy variable names and immutable raw outputs.

## What was frozen

The final run fits the local quadratic risk surface

\[
\hat r(\theta_0+u)
=\hat\tau+\hat b^\top u+\frac12u^\top\hat H u.
\]

The calibration center is

\[
\theta_0=(0.75,0.30,0).
\]

The observed error at the center environment was 0.433594. The fitted
quadratic intercept was

\[
\hat\tau=\hat r(\theta_0)=0.419018.
\]

The protocol froze `gamma = 0.04`, producing the relative comparison sets

\[
\mathcal H_-:\ r(u)\le\hat\tau-\gamma=0.379018,
\]

\[
\mathcal H_+:\ r(u)\ge\hat\tau+\gamma=0.459018.
\]

The asymmetric pair was designed with an additional frozen model-remainder
buffer of 0.02, hence `design_margin = 0.06`.

## What the experiment establishes

1. A quadratic deployment-error surface can be estimated stably on the local
   TorchSig calibration grid (`CV R2 = 0.971`).
2. An asymmetrically optimized least-observable pair satisfies the frozen risk
   and support constraints.
3. The mechanism-information expression predicts the sample-size dependence of
   lower-versus-higher-risk state discrimination (`MAE = 0.0230`,
   `Spearman = 0.941`).
4. The revealed point risks cross both frozen relative comparison cutoffs.
5. The conservative two-sided 95% interval rule abstains by 0.000292 on the
   lower-risk side. The final status is therefore 12/13 gates, not full pass.
6. In the Round 6 extension, a frozen 25-dimensional observation channel
   retains 84.3%–89.1% of full-channel worst-pair information at two local
   centers and passes all target-curve and mean accuracy-loss gates.
7. In the final Round 7 replay, the same 25-dimensional channel provides
   prospective information under persistent local drift: timely warning is
   64.4%, false alarm 11.1%, median lead three windows, and gain over current-
   state detection 51.7 percentage points.
8. The Round 7 result is 9/10 gates. Premature warning is 21.7%, above the
   frozen 20% maximum, so a full warning certificate is not supported.
9. Round 8 isolates the source of the prospective increment with a matched
   temporal-order knockout. Holding the exact state multiset and terminal state
   fixed, ordered histories exceed shuffled histories by 0.030075 in integrated
   forecast score (95% paired-bootstrap interval [0.027460, 0.032763]); 98.33%
   of pairs favor ordered histories.
10. The Round 8 result passes all 11 formal mechanism and integrity gates over
    120 fresh-seed matched replicates and 28,800 shuffle replays. Both tested
    directions are positive, while the sudden-jump proxy pre-warning rate is
    0.83%.
11. Round 9 shows that the deployed warning readout need not receive the named
    three-dimensional TorchSig mechanism coordinate. The formal 25D blind chart
    reaches 71.67% timely warning, zero stationary false alarms, median lead 2.5
    windows, and a minimum direction-specific timely rate of 61.67%.
12. The fixed training reference does not improve on a matched 25D moment
    chart. The timely-warning contrast is -0.00833 with paired-bootstrap
    interval [-0.025, 0]; the position-MAE contrast is effectively zero.
13. Round 10 controls the training distribution, retrains 20 models, and
    evaluates them on identical calibration and deployment samples. Every
    baseline path crosses the common Round 9 boundary and no five-percent
    Cliff-aware path crosses it.
14. Cliff-aware terminal risk is 0.3331 below baseline with a training-seed-
    clustered interval [-0.3683, -0.29045]. Against equal-budget random broad
    coverage, the terminal-risk advantage is 0.0120 with interval
    [0.00350, 0.02625] in favor of Cliff-aware training.
15. Cliff-aware training contracts the mean gradient norm to 0.50% and the
    Hessian Frobenius norm to 3.81% of baseline, converting the frozen challenge
    from a warning problem into a near-flat low-risk region.
16. The fixed-channel `trace(Q)` changes by -21.0027 with clustered interval
    [-23.1758, -18.8066], establishing that training affects observation
    geometry as well as the risk surface.
17. Round 11C gives exact finite-sample evidence that risk change equals
    forward minus recovery flux across the true decision boundary. First
    crossings are persistent and temporally distributed rather than a required
    synchronized pulse.
18. Across five fresh training seeds, Cliff-aware training reduces endpoint
    risk by 0.27285 (training-seed cluster 95% interval [0.21543, 0.33496]) and
    incident crossing by 0.12148 ([0.08574, 0.15176]).

## What the experiment does not establish

1. TorchSig supplies no accepted operational harm threshold corresponding to
   error rates 0.379018 or 0.459018.
2. The fitted intercept is a local statistical anchor, not an independently
   discovered physical phase transition.
3. The four-percentage-point `gamma` is a frozen effect size, not a value
   derived from utility, cost, or an engineering safety requirement.
4. Therefore the run does not certify absolute deployment safety or
   catastrophe. It validates a relative risk-separation and identifiability
   mechanism.
5. The measurement experiment does not establish that the Cliff-directed
   subset objective is superior to generic `trace(Q)` design. That designated
   claim was rejected at the primary budget because both objectives selected
   the same subset.
6. The 25-dimensional compression result is local to the frozen TorchSig model,
   centers, grouping, and scalar-dimension cost definition; hardware cost and
   cross-benchmark transfer remain untested.
7. Future crossing is identifiable only under the stated short-horizon drift-
   persistence condition. The experiment does not support warning under
   arbitrary, discontinuous, or adversarial dynamics.
8. The warning boundary is still protocol-relative, not an externally grounded
   engineering safety threshold. Warning therefore means predicted crossing of
   the frozen relative-risk boundary, not predicted catastrophe.
9. The Q90-buffered warning rule has one independent TorchSig replay. External
   hardware, another model family, and another benchmark remain necessary for
   a broad deployment claim.
10. The order knockout identifies temporal accumulation as the source of the
    trend rule's prospective increment only within the frozen local-drift
    intervention. It does not prove that every physical degradation process is
    gradual or that the sensor trace causally produces the future failure.
11. A shuffled history is a controlled mechanism knockout, not necessarily a
    naturally occurring deployment trajectory. Its value is conditional
    identification after current level and state inventory are held fixed.
12. The Round 8 confirmation covers two calibration-feasible directions in one
    TorchSig model. The phase-axis endpoint failure, the invalidated v2 rerun,
    and the NumPy-only TorchSig compatibility runtime are retained limitations.
13. Round 9 hides (u) from the deployed predictor, not from the controlled
    experiment generator. Labelled calibration environments remain necessary,
    and their construction in an uncontrolled real system remains open.
14. The wrong-reference swap demonstrates coordinate sensitivity, not positive
    incremental information from the correct training set. For the fixed model,
    matched reference-free telemetry moments perform at least as well.
15. Round 10's training-distribution conclusion is causal only within the
    controlled TorchSig simulator, frozen ExtraTrees family, and two deployment
    paths. External models and real RF hardware remain untested.
16. Cliff-aware enrichment uses the simulator's hidden mechanism coordinates.
    It does not solve latent mechanism identification when deployment controls
    are unavailable.
17. The common Round 9 boundary is protocol-relative, not an engineering harm
    threshold. Preventing its crossing is not an absolute safety certificate.
18. Both five-percent enrichment methods prevent crossing, so targeted
    superiority is established only on terminal-risk and risk-area endpoints.
19. When training makes `b` and `H` nearly vanish, risk-null ratios and inverse
    risk-coordinate information are scale-unstable. The observed change in
    `Q` is not evidence that a lower `trace(Q)` is universally desirable.
20. Five paired training seeds do not establish a population result over
    training algorithms, architectures, benchmarks, or distribution shifts.
21. Round 11A rejects synchronized first crossing as a necessary explanation in
    its paired smoke panel. Round 11B supports a persistent distributed-flux
    replacement in the disclosed approximate compatibility runtime.
22. Round 11C removes the approximate signal-kernel confound using the exact
    tagged TorchSig 2.1.1 algorithms exercised by this configuration, but it
    still uses a narrow NumPy import runtime rather than installed PyTorch and
    TorchSig packages. Package-runtime replay remains required.
23. Post-paper Round 12 smoke experiments do not establish a control-free
    targeted-repair advantage. Persistent crossing, queried signed flux, and
    one-at-a-time influence scores all fail to beat equal-budget random
    deployment coverage on fresh terminal risk. These stopped smokes do not
    alter the manuscript v5 mechanism claim.

## Legacy field-name mapping

Raw outputs are retained unchanged. Interpret their historical keys as follows:

| Legacy field | Correct interpretation |
| --- | --- |
| `risk_minus` / `safe_side` | Lower-risk state |
| `risk_plus` / `cliff_side` | Higher-risk state |
| `safe_cutoff` | Lower relative comparison cutoff |
| `cliff_cutoff` | Higher relative comparison cutoff |
| `rich_safe_cliff_realized` | Point estimates cross both relative cutoffs |
| `rich_safe_cliff_ci_realized` | Two-sided 95% intervals cross both relative cutoffs |

An absolute safe-versus-catastrophic claim requires an independently specified
operational boundary from a loss function, engineering requirement, or domain
standard, followed by a new sealed validation.
