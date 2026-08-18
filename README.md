# Cliff SBT reproducibility repository

Lightweight, GitHub-ready reproduction assets for **Generalization cliffs in classifiers emerge from persistent signed boundary transport** (manuscript v7).

This repository is deliberately tiered:

1. **Fast verification, no external datasets** — validate manuscript-facing numbers from compact committed evidence and regenerate all main/Extended Data figures.
2. **Method reuse** — install the bundled `sbt-monitor` source and compute identity-paired transport ledgers on new systems.
3. **Full experimental replay** — use the domain code under `experiments/` and obtain the omitted public datasets/weights or the full 145 MB evidence archive using the recorded SHA-256.

The lightweight repository excludes raw image archives, large feature tensors, checkpoints, caches, duplicate historical outputs, and exploratory target arrays. It retains final source code, frozen protocols/configurations, compact evidence, final paper text, reference figures, tests, and provenance manifests.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lite.txt
python -m pip install -e tooling/sbt-monitor[warning]

python reproduce/preflight_git_tracking.py
python reproduce/verify_compact_evidence.py
python reproduce/make_figures.py --evidence-dir evidence/compact --out-dir figures/rebuilt
python -m pytest tooling/sbt-monitor/tests -q
python reproduce/audit_repository.py --strict
```

Or run:

```bash
make preflight-git
make verify
make figures
make test
make audit
# or: make all
```

## Scientific scope

The repository preserves four distinct objects:

- **Task SBT ledger:** exact correct-to-error minus error-to-correct accounting; requires fixed-model, identity-paired outcomes.
- **Operational event rule:** a user-declared boundary plus a separately declared persistence rule.
- **Prediction-state transport proxy:** outcome-blind departure/return telemetry; it is not exact task SBT.
- **Warning readout:** domain-calibrated and conditional; no universal detector or safety threshold is shipped.

See `tooling/sbt-monitor/API_SCIENTIFIC_SCOPE_v0.1.md` and `docs/CLAIMS_AND_LIMITS.md`.

## Repository map

```text
paper/                  v7 main manuscript, SI and 200-word editorial abstract
figures/reference/      committed main and Extended Data figures (PNG + PDF)
reproduce/              compact-evidence verification and figure regeneration
evidence/compact/       small committed tables sufficient for fast checks/figures
experiments/            cleaned formal source code and frozen protocols by domain
tooling/sbt-monitor/    installable ledger-first Python package source
docs/                   evidence map, data notice and full-replay instructions
```

## What is and is not reproduced

`verify_compact_evidence.py` checks the exact paired accounting and key manuscript-facing results from committed summary/row-level tables. It does not retrain models. Full retraining requires domain-specific environments and public source data described in `docs/REPRODUCIBILITY.md`.

## Anonymity and release metadata

Authorship metadata is intentionally generic for double-blind use. Replace the organization entry in `CITATION.cff` and add the permanent archival DOI before a named public release.
