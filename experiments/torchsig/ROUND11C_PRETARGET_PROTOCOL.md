# Round 11C pretarget protocol: source-faithful boundary flux

## Question

Does the Round 11B replacement mechanism survive removal of the approximate
NumPy signal kernel?

The frozen mechanism is

\[
R_{t+1}-R_t = \{F_t-B_t\}/N,
\]

where `F_t` is forward true-decision-boundary crossing and `B_t` is recovery.
A Cliff is cumulative net boundary flux exhausting the available risk
headroom. A synchronized crossing pulse is neither required nor targeted.

## Runtime scope

The signal algorithms are transcribed from the TorchSig `v2.1.1` tag at
commit `d9abfe1af2b0216d2bacc31c677407ed31878086`: the constellation maps,
SRRC pulse shaping, unit-rate constellation modulator, nonlinear amplifier,
carrier phase noise, and AWGN used by the frozen generator. The experiment's
`sample_rate / bandwidth / 4` resampling factor is exactly one.

This is an official-source NumPy execution, not an installed standard
PyTorch/TorchSig package runtime. A pass cannot be labeled package-runtime
confirmation.

## Frozen design

- Fresh paired panel: 512 examples per path.
- Same latent symbol, phase-noise and AWGN variates across every time point.
- Two frozen TorchSig paths: noise and mixed-gradient.
- Five fresh training seeds, each pairing baseline and 5% Cliff-aware models.
- 160 training samples per class and 140 ExtraTrees per model.
- Relative operational boundary: anchor risk plus 0.04, confirmed for two windows.
- Inference: two paths are averaged within training seed, then training seeds are
  resampled in a 20,000-draw cluster bootstrap.

## Frozen tests

1. Exact finite-sample flux accounting.
2. Baseline operational cliffs on at least 80% of model-path pairs.
3. Persistent incident crossings.
4. High crossing-time entropy and no dominant three-window pulse.
5. No requirement for extra position-velocity coupling.
6. Specificity to the true model boundary against random-boundary placebos.
7. Cliff-aware reduction in endpoint risk and incident crossings.
8. Both training-seed cluster interval lower bounds are strictly positive.
9. Both paths have positive median endpoint and incident-crossing reductions.

All thresholds, seeds, source files and analysis code are hashed before target
execution. Failed gates remain failed; no posttarget threshold repair is allowed.
