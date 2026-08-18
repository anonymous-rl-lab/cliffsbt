# Round 13: second neural-network mechanism domain

This directory implements the staged qualification of a second paired image
domain for the Cliff boundary-transport mechanism.

## Scientific status

- **CURE-TSR is the intended formal external field.** Its public documentation
  defines challenge types, five ordered challenge levels, and an instance index,
  but does not explicitly guarantee that an index denotes the same underlying
  image across levels. The data also require an author-provided download link.
- **The local CIFAR-10 experiment is a qualification smoke, not an official
  CIFAR-10-C benchmark result.** It applies three deterministic, ordered
  corruptions to the same CIFAR-10 test identities and asks whether paired
  decision-boundary accounting transfers to a trained convolutional network.

The staged series is now complete through the official benchmark:

- Round 13B controlled smoke: **8/8, ADVANCE**;
- Round 13C multi-seed pilot: **9/9, ADVANCE**;
- Round 13D official CIFAR-10-C: **12/12, MECHANISM_DOMAIN_CONFIRMED**.
- Round 13E equal-budget repair: **6/9, PARTIAL_OR_STOP**. Coverage repaired
  baseline and beat random augmentation, but did not beat equally dangerous,
  deeper first-crossing samples.

See `ROUND13_SERIES_REPORT.md` for the scientific interpretation. The result
now establishes repair efficacy in this controlled image domain, but rejects
coverage-only dominance and does not establish CURE-TSR identity pairing.

## Reproduction

```bash
../.venv_round13/bin/python src/audit_cure_pairing.py --root /path/to/CURE-TSR
../.venv_round13/bin/python src/run_cifar10_paired_smoke.py \
  --data-root data --out results/cifar10_paired_smoke_seed13 \
  --seed 13 --train-size 10000 --test-size 2000 --epochs 4

../.venv_round13/bin/python src/run_cifar10_multiseed_pilot.py \
  --data-root data --out results/cifar10_multiseed_pilot_v1

../.venv_round13/bin/python src/run_cifar10c_official.py \
  --data-root data/CIFAR-10-C --archive data/CIFAR-10-C.tar \
  --checkpoint-dir results/cifar10_multiseed_pilot_v1 \
  --out results/cifar10c_official_v1

../.venv_round13e_cpu/bin/python src/run_round13e_repair.py \
  --mode formal --data-root data \
  --round13d-outputs results/cifar10c_official_v1/paired_outputs.npz \
  --out results/round13e_formal_v1 --threads 4 --batch-size 256

../.venv_round13/bin/python audit_round13.py
../.venv_round13e_cpu/bin/python audit_round13e.py
```

The run writes the frozen configuration, checkpoint, per-level metrics, paired
raw predictions, and a machine-readable gate decision.

Source images are intentionally excluded from the distributable package; see
`DATA_NOTICE.md`.
