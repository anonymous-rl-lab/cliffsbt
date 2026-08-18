# Claim-to-artifact map

| Manuscript component | Fast evidence | Formal source |
|---|---|---|
| Paired TorchSig formation | `evidence/compact/torchsig_round11c_paired_effects.csv` | `experiments/torchsig/src/run_round11c_official_source_flux.py` |
| Temporal-order intervention | `evidence/compact/torchsig_temporal_order_pairs.csv` | `experiments/torchsig/src/probe_precursor_order_knockout.py` |
| Training-support control | Round 10 trajectory/geometry tables and decomposition CSV | `experiments/torchsig/src/run_round10_training_intervention.py` |
| Sparse repair contrast | Round 12C compact tables | `experiments/torchsig/src/run_round12c_coverage_vs_hazard_pilot.py` |
| CIFAR-10-C formation | `cifar10c_endpoint_by_seed.csv` plus diagnostic placebos | `experiments/cifar10c/src/run_cifar10c_official.py` |
| CIFAR repair reversal | compact selection and seed-result tables | `experiments/cifar10c/src/run_round13e_repair.py` |
| CURE-OR serial loop | compact path/seed tables | `experiments/cure_or/code/run_phase1.py`, `run_phase2.py` |
| Warning comparisons | `evidence/compact/v6_diagnostics/` and `v7_diagnostics/` | diagnostic builders retained in the paper analysis provenance |
| Figures | all compact tables | `reproduce/make_figures.py` |
