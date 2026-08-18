#!/usr/bin/env python3
"""Verify that every file named by MANIFEST.sha256 is actually tracked by Git.

This catches a class of release errors that an ordinary working-tree audit cannot:
a file may exist in a ZIP, yet be omitted from GitHub because it matches .gitignore.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--require-git",
        action="store_true",
        help="Fail instead of skipping when the directory is not a Git worktree.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    probe = run_git(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        result = {
            "root": str(root),
            "status": "FAIL" if args.require_git else "SKIP",
            "reason": "not_a_git_worktree",
            "problems": [],
        }
        print(json.dumps(result, indent=2))
        return 1 if args.require_git else 0

    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        print(
            json.dumps(
                {
                    "root": str(root),
                    "status": "FAIL",
                    "problems": ["missing:MANIFEST.sha256"],
                },
                indent=2,
            )
        )
        return 1

    manifest_paths: list[str] = []
    duplicate_paths: list[str] = []
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            _digest, rel = line.split("  ", 1)
        except ValueError:
            print(
                json.dumps(
                    {
                        "root": str(root),
                        "status": "FAIL",
                        "problems": [f"malformed_manifest_line:{line}"],
                    },
                    indent=2,
                )
            )
            return 1
        if rel in seen:
            duplicate_paths.append(rel)
        seen.add(rel)
        manifest_paths.append(rel)

    tracked_proc = run_git(root, "ls-files", "-z")
    if tracked_proc.returncode != 0:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "status": "FAIL",
                    "problems": ["git_ls_files_failed"],
                    "stderr": tracked_proc.stderr.strip(),
                },
                indent=2,
            )
        )
        return 1
    tracked = {item for item in tracked_proc.stdout.split("\0") if item}

    missing_on_disk = [rel for rel in manifest_paths if not (root / rel).is_file()]
    untracked = [rel for rel in manifest_paths if rel not in tracked]
    ignored: list[dict[str, str]] = []
    for rel in untracked:
        proc = run_git(root, "check-ignore", "-v", "--no-index", "--", rel)
        if proc.returncode == 0 and proc.stdout.strip():
            ignored.append({"path": rel, "rule": proc.stdout.strip()})

    problems: list[str] = []
    problems.extend(f"manifest_duplicate:{rel}" for rel in duplicate_paths)
    problems.extend(f"manifest_missing:{rel}" for rel in missing_on_disk)
    problems.extend(f"manifest_untracked:{rel}" for rel in untracked)

    result = {
        "root": str(root),
        "manifest_entries": len(manifest_paths),
        "tracked_files": len(tracked),
        "problems": problems,
        "ignored_details": ignored,
        "status": "PASS" if not problems else "FAIL",
    }
    print(json.dumps(result, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
