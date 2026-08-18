#!/usr/bin/env python3
"""Build non-circular per-run and repository SHA-256 manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "SHA256SUMS.txt"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_manifest(base: Path, output: Path) -> None:
    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and path != output
        and "__pycache__" not in path.parts
        and ".venv" not in path.parts
    )
    lines = [f"{digest(path)}  {path.relative_to(base)}" for path in files]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = ROOT / "results"
    for run_dir in sorted(path for path in results.iterdir() if path.is_dir()):
        write_manifest(run_dir, run_dir / MANIFEST_NAME)
    write_manifest(ROOT, ROOT / MANIFEST_NAME)


if __name__ == "__main__":
    main()

