#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
M=json.loads((ROOT/'evidence'/'full_evidence_manifest.json').read_text())
P=argparse.ArgumentParser(); P.add_argument('--url',default=M.get('download_uri')); P.add_argument('--out',type=Path,default=ROOT/'external' / M['archive_name']); A=P.parse_args()
if not A.url: raise SystemExit('No archival URL is frozen yet. Supply --url after the full archive is deposited.')
A.out.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve(A.url,A.out)
h=hashlib.sha256(A.out.read_bytes()).hexdigest()
if h!=M['sha256']: A.out.unlink(missing_ok=True); raise SystemExit(f'SHA-256 mismatch: {h}')
print(A.out)
