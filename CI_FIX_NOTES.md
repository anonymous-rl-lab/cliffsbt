# GitHub Actions tracking fix — repository patch 1.0.1

## Root cause

`MANIFEST.sha256` correctly listed `experiments/cure_or/audit/PHASE1_EXECUTION.log`, but the original `.gitignore` ignored every `*.log` file. The log existed in the distributed ZIP and therefore passed the local working-tree audit, yet a normal `git add .` omitted it. After `actions/checkout`, the manifest referenced a file that was not present, causing the Python 3.10 strict audit to fail.

The Python 3.12 matrix job did not independently fail: GitHub Actions cancelled it because matrix `fail-fast` was enabled after the Python 3.10 job failed.

## Patch

1. Added a `.gitignore` exception for the frozen CURE-OR execution log.
2. Added `reproduce/preflight_git_tracking.py`, which verifies that every manifest entry is tracked by Git.
3. Added the tracking preflight to GitHub Actions.
4. Set `strategy.fail-fast: false` so both supported Python versions finish independently.
5. Set `MPLBACKEND=Agg` explicitly for headless figure generation.
6. Regenerated the repository manifest and package inventory.

## Patch for an already-created GitHub checkout

```bash
printf '\n!experiments/cure_or/audit/PHASE1_EXECUTION.log\n' >> .gitignore
git add .gitignore
git add -f experiments/cure_or/audit/PHASE1_EXECUTION.log
git add .github/workflows/ci.yml reproduce/preflight_git_tracking.py
git commit -m "Fix tracked evidence log and CI matrix"
git push
```

Using the corrected package is preferable because it also contains the regenerated manifest and preflight documentation.
