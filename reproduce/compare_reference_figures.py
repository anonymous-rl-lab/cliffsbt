#!/usr/bin/env python3
"""Byte-compare regenerated PNG figures with committed reference PNGs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
P=argparse.ArgumentParser()
P.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[1]/'figures'/'reference')
P.add_argument('--rebuilt',type=Path,default=Path(__file__).resolve().parents[1]/'figures'/'rebuilt')
A=P.parse_args(); failures=[]; rows=[]
for ref in sorted(A.reference.rglob('*.png')):
    rel=ref.relative_to(A.reference); got=A.rebuilt/rel
    rh=hashlib.sha256(ref.read_bytes()).hexdigest(); gh=hashlib.sha256(got.read_bytes()).hexdigest() if got.exists() else None
    ok=rh==gh; rows.append({'figure':rel.as_posix(),'match':ok,'reference_sha256':rh,'rebuilt_sha256':gh})
    if not ok: failures.append(rel.as_posix())
print(json.dumps({'figures':len(rows),'matched':len(rows)-len(failures),'failures':failures},indent=2))
raise SystemExit(bool(failures))
