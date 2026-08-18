# Phase 2 execution note

- Parent prospective registration: <https://osf.io/c6ygf>
- Associated-project Phase 1 evidence: <https://osf.io/nm3ex/files/x8vmb>
- Phase 2 execution time: 2026-08-17 13:40:31 Asia/Singapore
- Protocol version: 2.0
- Python: 3.12.13
- NumPy: 2.5.2
- PyTorch: 2.13.0+cpu
- torchvision: 0.28.0+cpu
- Pillow: 12.3.0
- scikit-learn: 1.9.0
- Registered-package strict audit: PASS, 45 files
- Phase 1 independent audit: PASS, 22/22 checks
- Synthetic Phase 2 end-to-end smoke test: PASS
- Formal command exit status: 0
- Blind commitment preserved: true
- Phase order check: true

Registered command:

```bash
PYTHONPATH=code python code/run_phase2.py \
  --workspace /absolute/path/CURE_OR_V2_PHASE2_FORMAL_x8vmb
```

Machine output:

```json
{
  "overall_decision": "FORMATION_WARNING_REPAIR_CONFIRMED",
  "h1_pass": true,
  "h2_pass": true,
  "h3_pass": true
}
```

No registered scientific input, decision rule, or committed Phase 1 file was
changed in this execution.
