# Reproducibility guide

## Environment used for the final mechanism sequence

- Python 3.12.13
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0

The original applicability smoke additionally requires PyTorch. Later mechanism runs
use the scikit-learn MLP and do not require PyTorch.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

For the original Torch MLP source smoke:

```bash
python -m pip install -r requirements-torch.txt
```

## Data

The exact raw UCI gzip and scikit-learn cache are included. Verify source hashes against
`DATA_NOTICE.md` and then verify the complete package:

```bash
sha256sum -c integrity/MANIFEST.sha256
```

## Recommended verification order

Do not run every directory as if it were one homogeneous preregistration. Several
directories are explicitly posttarget diagnostics. Read each local `PROTOCOL.md` and
`config.json` first.

Terminal registered stages can be checked with:

```bash
python work_covtype_45m_multiseed_pilot/run_pilot.py freeze
python work_covtype_45m_formal_v1/run_formal.py freeze
python work_covtype_robust_margin_holdout/run_holdout.py freeze
```

Expected hashes:

- 45 m pilot: `0e4ad5ba865739c64978221cf73e541227bf59cc777643afef69d3461971887f`
- 45 m formal: `ee449031ed004279277e2fdd52781e859fd5be9b92543e07c0d78a55a6bbed92`
- final fresh holdout: `a015c3c36fccb92e8c81c9d846e5f3a16d6c3614321139e3fc326b9b13b69aad`

To rerun a frozen stage, pass its expected hash exactly as specified by its runner. A
full rerun overwrites only that stage's `results/` files; preserve the packaged copy
before doing so.

## Important scope

Rerunning the code reproduces the computational protocol. It does not make diagnostic
scale selection prospective, restore the unavailable historical C1/C2 directories, or
turn unpaired Covertype windows into longitudinal samples.
