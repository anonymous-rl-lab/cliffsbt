# Round 13D: official CIFAR-10-C paired mechanism benchmark

Frozen after the Round 13C controlled pilot passed 9/9 gates and before the
official CIFAR-10-C archive was extracted or evaluated.

## Purpose and claim boundary

Round 13D asks whether the paired formation law transfers to the official
CIFAR-10-C common-corruption benchmark. CIFAR-10-C is a controlled image domain,
not the final real-world CURE-TSR validation. Passing Round 13D would establish a
second neural-network mechanism domain, not universal Cliff behavior.

## Fixed assets

- official CIFAR-10-C v1 archive, DOI 10.5281/zenodo.2535967;
- required archive MD5: `56bf5dcef84df0e2308c6dcbcbbd8499`;
- three frozen Round 13C checkpoints: training seeds 31, 47, 61;
- all 10,000 CIFAR-10 test identities in official order;
- standard 15 corruption families, five official severities;
- clean CIFAR-10 test set as level 0;
- operational threshold: each model's clean error + 0.15;
- training seed is the uncertainty cluster; 10,000 bootstrap draws.

Standard families:

`gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`,
`glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `frost`, `fog`,
`brightness`, `contrast`, `elastic_transform`, `pixelate`, and
`jpeg_compression`.

## Frozen controls

1. Exact paired identity layout: each corruption array must contain 5 blocks of
   10,000 images; `labels.npy` must contain five exactly repeated 10,000-label
   blocks.
2. Exact true-boundary transition accounting at every adjacent severity.
3. Fixed incorrect pseudo-label boundary placebo.
4. Common-stream evaluation of all three pre-trained models.
5. No corruption family or severity may be removed after reveal.

## Frozen gates

| Gate | Threshold |
|---|---:|
| Archive integrity | official MD5 exact |
| Official identity layout | 15/15 arrays and labels have 50,000 entries; five label blocks are identical |
| Model competence | clean accuracy >= 0.45 for 3/3 seeds |
| Exact task-boundary accounting | maximum absolute error <= 1e-12 |
| Reproducible headroom exhaustion | >=10/15 families cross for all three seeds |
| First-crossing correctness | cumulative flux is below headroom before, and reaches it at, every declared crossing |
| Persistent crossings | cluster-bootstrap lower 95% bound > 0.70 among crossing cells |
| Distributed crossings | cluster-bootstrap lower 95% bound of normalized first-crossing entropy > 0.50 |
| Boundary-local ordering | cluster-bootstrap lower 95% bound of prior-margin gap > 0 |
| Boundary specificity | cluster-bootstrap lower 95% bound of placebo-minus-true accounting RMSE > 0.01 |
| Seed-direction consistency | endpoint risk growth is positive for all seeds on >=12/15 families |
| Cross-family heterogeneity | median cliff level occupies >=2 distinct levels |

All twelve gates are required. Failure is retained as a bounded result; the
protocol is not repaired after target reveal.
