#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
P=argparse.ArgumentParser(); P.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); P.add_argument('--strict',action='store_true'); A=P.parse_args(); root=A.root.resolve()
problems=[]
required=['README.md','README_ZH.md','LICENSE','CITATION.cff','MANIFEST.sha256','paper/Cliff_NMI_v7.md','paper/Cliff_NMI_Supplementary_v7.md','evidence/compact/cure_or_path_level_results.csv','tooling/sbt-monitor/pyproject.toml','reproduce/verify_compact_evidence.py','reproduce/make_figures.py','reproduce/preflight_git_tracking.py']
for r in required:
    if not (root/r).exists(): problems.append('missing:'+r)
# no accidental heavy objects or local absolute paths
for p in root.rglob('*'):
    if not p.is_file() or '.git' in p.parts: continue
    rel=p.relative_to(root).as_posix()
    if p.stat().st_size>12_000_000: problems.append(f'oversized:{rel}:{p.stat().st_size}')
    if rel != 'reproduce/audit_repository.py' and not rel.startswith('docs/source_archive/') and p.suffix.lower() in {'.py','.md','.json','.yml','.yaml','.toml','.txt','.csv'} and p.stat().st_size<2_000_000:
        txt=p.read_text(errors='ignore')
        if '/mnt/data/' in txt or '/root/autodl' in txt: problems.append('absolute_path:'+rel)
        if re.search(r"(?i)(api[_-]?key|token|password)\s*[=:]\s*[\"'][A-Za-z0-9_\-]{12,}", txt): problems.append('possible_secret:'+rel)
# manifest
manifest=root/'MANIFEST.sha256'
if manifest.exists():
    for line in manifest.read_text().splitlines():
        if not line.strip(): continue
        h,rel=line.split('  ',1); p=root/rel
        if not p.exists(): problems.append('manifest_missing:'+rel); continue
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=h: problems.append('manifest_mismatch:'+rel)
print(json.dumps({'root':str(root),'problems':problems,'status':'PASS' if not problems else 'FAIL'},indent=2))
raise SystemExit(bool(problems) and A.strict)
