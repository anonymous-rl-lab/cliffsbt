#!/usr/bin/env python3
"""Self-contained release audit for sbt-monitor v0.1.0."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition: bool, message: str) -> dict[str, object]:
    return {"pass": bool(condition), "message": message}


def main() -> int:
    checks: list[dict[str, object]] = []
    source_spec = ROOT / "API_SCIENTIFIC_SCOPE_v0.1.md"
    package_spec = ROOT / "src/sbt_monitor/data/API_SCIENTIFIC_SCOPE_v0.1.md"
    checks.append(check(source_spec.read_bytes() == package_spec.read_bytes(), "bundled specification matches root specification"))

    init_text = (ROOT / "src/sbt_monitor/__init__.py").read_text(encoding="utf-8")
    checks.append(check("WarningCalibrator" not in init_text, "experimental warning calibrator is not top-level"))
    proxy_text = (ROOT / "src/sbt_monitor/proxy.py").read_text(encoding="utf-8")
    checks.append(check("net_sbt" not in proxy_text.lower(), "online proxy implementation does not use prohibited net_sbt name"))

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    checks.append(check(test.returncode == 0, "source test suite passes"))

    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
    checks.append(check(len(wheels) == 1, "exactly one wheel exists"))
    checks.append(check(len(sdists) == 1, "exactly one sdist exists"))

    metadata_summary: dict[str, object] = {}
    if wheels:
        with zipfile.ZipFile(wheels[0]) as archive:
            bad = archive.testzip()
            checks.append(check(bad is None, "wheel ZIP integrity passes"))
            meta_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = Parser().parsestr(archive.read(meta_name).decode("utf-8"))
            metadata_summary = {
                "name": metadata.get("Name"),
                "version": metadata.get("Version"),
                "requires_python": metadata.get("Requires-Python"),
                "license_expression": metadata.get("License-Expression"),
                "description_content_type": metadata.get("Description-Content-Type"),
            }
            checks.append(check(metadata.get("Name") == "sbt-monitor", "wheel name is sbt-monitor"))
            checks.append(check(metadata.get("Version") == "0.1.0", "wheel version is 0.1.0"))
            checks.append(check(metadata.get("License-Expression") == "Apache-2.0", "wheel license expression is Apache-2.0"))
            checks.append(check("sbt_monitor/data/API_SCIENTIFIC_SCOPE_v0.1.md" in archive.namelist(), "wheel contains frozen specification"))
    if sdists:
        with tarfile.open(sdists[0]) as archive:
            names = archive.getnames()
            checks.append(check(any(name.endswith("/tests/test_ledger.py") for name in names), "sdist contains tests"))
            checks.append(check(any(name.endswith("/docs/SCIENTIFIC_SCOPE.md") for name in names), "sdist contains documentation"))

    report = {
        "package": "sbt-monitor",
        "version": "0.1.0",
        "spec_sha256": sha256(source_spec),
        "metadata": metadata_summary,
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(DIST.iterdir())
            if path.is_file()
        },
        "checks": checks,
        "test_stdout": test.stdout,
        "test_stderr": test.stderr,
        "all_pass": all(item["pass"] for item in checks),
    }
    destination = ROOT / "RELEASE_AUDIT.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
