PYTHON ?= python
EVIDENCE := evidence/compact

.PHONY: preflight-git verify figures figures-check test audit all
preflight-git:
	$(PYTHON) reproduce/preflight_git_tracking.py

verify:
	$(PYTHON) reproduce/verify_compact_evidence.py --evidence-dir $(EVIDENCE)
figures:
	$(PYTHON) reproduce/make_figures.py --evidence-dir $(EVIDENCE) --out-dir figures/rebuilt
figures-check: figures
	$(PYTHON) reproduce/compare_reference_figures.py

test:
	PYTHONPATH=tooling/sbt-monitor/src $(PYTHON) -m pytest tooling/sbt-monitor/tests -q
audit:
	$(PYTHON) reproduce/audit_repository.py --strict
all: preflight-git verify test figures-check audit
