# Round 11C: TorchSig 2.1.1 official-source boundary-flux confirmation

## Result

Round 11C passed all 12 frozen gates. It used the exact TorchSig `v2.1.1`
signal algorithms needed by the experiment, executed through a narrow NumPy
runtime, five fresh training seeds, and a fresh 512-sample paired panel on each
of the two frozen deployment paths.

The result confirms the Round 11B replacement mechanism at the signal-kernel
level:

> The observed operational Cliff is accumulated, persistent net transport of
> samples across the model's true decision boundary. The transport is spread
> over time; a synchronized crossing pulse and an extra position-velocity
> interaction are not necessary.

## Frozen results

| Quantity | Result |
| --- | ---: |
| Baseline relative-boundary crossing fraction | 1.0000 |
| Maximum flux-accounting error | 0.0000 |
| Median incident-crossing endpoint persistence | 0.9525 |
| Median normalized first-crossing entropy | 0.8789 |
| Median largest three-window incident share | 0.3827 |
| Baseline pairs with extra velocity coupling | 0/10 |
| Baseline pairs above random-boundary q95 | 9/10 |
| Mean Cliff-aware endpoint-risk reduction | 0.2729 |
| Training-seed cluster 95% interval | [0.2154, 0.3350] |
| Mean Cliff-aware incident-crossing reduction | 0.1215 |
| Training-seed cluster 95% interval | [0.0857, 0.1518] |
| Minimum pathwise median endpoint reduction | 0.2285 |
| Minimum pathwise median incident reduction | 0.1191 |

The pretarget manifest digest was
`e4bb820525969d856da9d2a700a46172915bd3cfc18159526a138dc27e03eddb`.
The paired panel digest was
`d360fd48701e7d4539be9943fc3dbb9491c9aeb841136450e673fbc318c23178`.

## Mechanistic interpretation

For the paired sample panel,

\[
R_{t+1}-R_t = \frac{F_t-B_t}{N}
\]

holds exactly, where `F_t` counts forward true-boundary crossings and `B_t`
counts recoveries. High first-crossing entropy, a small largest-three-window
share, and 95.25% endpoint persistence jointly show gradual accumulation of
mostly irreversible crossings. The velocity-shuffle result rejects the
stronger claim that boundary-near samples must receive an additional harmful
velocity. The random-boundary placebo shows that the observed flux is largely
specific to the trained model's true classification boundary.

Round 10's intervention now has a sample-level mechanism: Cliff-aware training
does not merely lower an endpoint metric. It reduces the number of samples
transported into the wrong-decision region and leaves fewer of them there.

## Runtime boundary

The environment could not install the standard PyTorch/TorchSig wheels. The
run therefore transcribed the tagged TorchSig algorithms for constellation
maps, SRRC shaping, unit-rate modulation, nonlinear amplification, carrier
phase noise, and AWGN. The frozen signal configuration has ideal resampling
rate exactly one, so no unimplemented resampling branch was used.

This is stronger than the approximate Round 11A/11B compatibility probes but
is not an installed-package runtime confirmation. The decision is therefore
`OFFICIAL_SOURCE_NUMPY_CONFIRMATION_PASS_PACKAGE_RUNTIME_REQUIRED`. The result
may enter the repository's mechanistic evidence ledger with this qualifier; a
standard-wheel replay remains the final runtime audit.

Upstream audit links:

- `https://pypi.org/project/torchsig/2.1.1/`
- `https://github.com/TorchDSP/torchsig/blob/v2.1.1/torchsig/signals/builders/constellation.py`
- `https://github.com/TorchDSP/torchsig/blob/v2.1.1/torchsig/signals/builders/constellation_maps.py`
- `https://github.com/TorchDSP/torchsig/blob/v2.1.1/torchsig/transforms/functional.py`
- `https://github.com/TorchDSP/torchsig/blob/v2.1.1/torchsig/utils/dsp.py`

## Evidence files

- `results/round11c_official_source_flux/summary.json`
- `results/round11c_official_source_flux/path_summary.csv`
- `results/round11c_official_source_flux/paired_effects.csv`
- `results/round11c_official_source_flux/transition_flux.csv`
- `results/round11c_official_source_flux/sample_first_crossings.csv`
- `results/round11c_official_source_flux/paired_margin_trajectories.npz`
- `figures/round11c_official_source_flux/round11c_official_source_flux.png`
