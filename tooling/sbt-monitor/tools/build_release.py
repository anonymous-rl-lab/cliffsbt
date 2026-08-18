#!/usr/bin/env python3
"""Build wheel and sdist and write SHA256SUMS without requiring a frontend."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    for egg in (ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir()
    cwd = Path.cwd()
    try:
        import os

        os.chdir(ROOT)
        build_sdist(str(DIST))
        build_wheel(str(DIST))
    finally:
        os.chdir(cwd)
    artifacts = sorted(path for path in DIST.iterdir() if path.is_file())
    lines = [f"{digest(path)}  {path.name}" for path in artifacts]
    (DIST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
