# 167 — `Voyage/` run-dir name in neither `.dockerignore` nor `.gitignore` (narrow gap beyond 069/108)

- **Severity:** LOW (nil impact today — the dir is empty, 4.0K; filed only because it is the one ignore gap neither 069 nor 108 lists)
- **File:line:** `Voyage/.dockerignore:1-19` (no `Voyage/` entry); `Voyage/.gitignore:1-16` (no `Voyage/` entry); `Voyage/Voyage/` (exists, root-owned, empty today)
- **Area:** containers / repo hygiene — ignore-file coverage

## Description

069 already covers `.coverage`/`coverage.xml`/`.pytest_cache`/`.hypothesis`/`.venv`/`*.egg-info`/`.env` gaps and 108 covers the stray root-owned `Voyage/Voyage/` dir's existence — but neither names the ignore gap: a misdirected `--output Voyage/...` (the exact failure mode both ignore-file headers describe — "a misdirected --output or local probe must never inflate the build context / commit", `.dockerignore:10-12`, `.gitignore:7-9`) would both commit run artifacts (jsonl/toml/segments) and inflate the build context, since `output/`-style coverage does not extend to the `Voyage/` name.

Boundaries (no overlap):

- 069 = context-bloat entries + stale scope comment (`.dockerignore:1-19` full-file audit); it lists artifact extensions and dot-caches but never the `Voyage/` directory name.
- 108 = the residue dir itself (`Voyage/Voyage/`, root-owned empty, git-invisible, `rmdir` fix); it records the anomaly class, not the ignore rule.
- This file = the single missing ignore line in each file (one line each, mirroring the `output/` entries). Fixing 069 + 108 as written still leaves `Voyage/state.json` committable and context-inflating.

## Live evidence (verified 2026-09-30)

```
$ ls -la Voyage/Voyage/ && du -sh Voyage/Voyage && stat -c "%U %a %y" Voyage/Voyage/
drwxr-xr-x 2 root root 4096 Sep 29 22:03 ./.   # empty — 108's dir, zero content today
4.0K Voyage/Voyage
root 755 2026-09-29 22:03:33
$ grep -c "Voyage/" Voyage/.dockerignore; grep -c "Voyage/" Voyage/.gitignore
0 / 0   # no entry (bare "Voyage" hits are only the header comments: .dockerignore:10 + .gitignore:8 "at the Voyage root")
$ grep -n "^Voyage" Voyage/.dockerignore Voyage/.gitignore || echo "no ^Voyage entries"
no ^Voyage entries
$ cat Voyage/.dockerignore  # 19 lines, has output/ but no Voyage/
$ cat Voyage/.gitignore     # 16 lines, has output/ but no Voyage/
$ ls Voyage/output/boba-probe/   # the same run shape that WOULD bloat/commit under Voyage/
concepts.jsonl logs novelty run_manifest.json segments state.json state.json.lock voyage.toml
```

## Repro

```bash
grep -c "Voyage/" Voyage/.dockerignore Voyage/.gitignore  # 0 / 0 — gap
ls -la Voyage/Voyage/  # empty today, so nil impact until a misdirected output lands there
mkdir -p Voyage/Voyage && touch Voyage/Voyage/state.json && git status --short  # shows (unignored)
docker build --progress=plain . 2>&1 | grep "transferring context"  # grows vs baseline
rm Voyage/Voyage/state.json  # leave the dir as found (108 owns the rmdir)
```

## Fix candidates

1. Add `Voyage/` to both files (one line each, mirroring the `output/` entries at `.dockerignore:5` / `.gitignore:6`).
2. Optionally fix the root ownership while there (`rmdir Voyage/Voyage/` — 108's remit, host-owned parent permits it).
3. Do NOT broaden to a `*Voyage*` glob — the literal `Voyage/` directory entry is exact and avoids ignore-rule surprises.

## Refs

- `Voyage/.dockerignore:1-19`; `Voyage/.gitignore:1-16`; `Voyage/Voyage/` (directory)
- `Voyage/issues/069_dockerignore_gaps.md` (context-bloat gaps — no `Voyage/` line); `Voyage/issues/108_stray_root_owned_nested_dir.md` (residue dir — no ignore rule)

(End of file)
