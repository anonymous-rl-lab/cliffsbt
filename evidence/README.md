# Evidence tiers

`compact/` contains row-level and summary tables sufficient for fast manuscript-number checks and figure regeneration. It does not contain raw image archives, model checkpoints, caches or the 35 MB CURE-OR feature tensor.

`full_evidence_manifest.json` records the SHA-256 of the omitted complete archive. `full_archive_SHA256SUMS.txt` preserves its internal file-level hash map.
