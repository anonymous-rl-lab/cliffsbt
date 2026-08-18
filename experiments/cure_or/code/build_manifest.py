#!/usr/bin/env python3
"""Build the non-circular SHA-256 manifest for this reproducibility package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "PACKAGE_MANIFEST.sha256"
AUDIT_REPORT = PACKAGE / "PACKAGE_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if MANIFEST.exists() and not args.force:
        raise SystemExit("manifest exists; pass --force to rebuild")
    excluded = {MANIFEST, AUDIT_REPORT}
    files = sorted(
        path for path in PACKAGE.rglob("*")
        if path.is_file() and path not in excluded and "__pycache__" not in path.parts
    )
    MANIFEST.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PACKAGE).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    print(f"MANIFEST BUILT: {len(files)} files; SHA-256 {sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
