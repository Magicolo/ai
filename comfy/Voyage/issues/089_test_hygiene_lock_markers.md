# 089 — Test hygiene + lock + markers: caches, lock typo, slow marker, mypy scope

- Severity: MEDIUM (tests / toolchain)
- Files: `.coverage` (52K SQLite, mtime 2026-09-25), `.hypothesis/` (348K), `.mypy_cache/` (51M, 3.10+3.12+CACHEDIR.TAG), `.ruff_cache/` (172K, 0.16.5/0.16.8/0.16.9), `requirements.lock` (55L), `pyproject.toml:182-203`, `scripts/gates.sh`, `scripts/build.sh`
- Area: tests / toolchain hygiene

## Description

Stale caches in tree (all gitignored but present): `.coverage` contradicts `data_file=/tmp/voyage-coverage-data`; `.hypothesis` materializes despite `conftest.py database=None`; `.mypy_cache` dual-version residue (scripts already export `MYPY_CACHE_DIR=/tmp/...`); `.ruff_cache` three versions (scripts export `RUFF_CACHE_DIR=/tmp/...`). `requirements.lock:25-26` `httpcore2==2.13.1`/`httpx2==2.13.1` — no such PyPI 2.x line (real is `httpx 0.28`); likely freeze-record typo — lock lies. `tomli` conditional (`python_version<'3.11'`) absent from lock (expected on 3.12 freeze host, 3.10 path unfrozen). Markers `gpu` + `endurance` registered (1 test each: `test_acestep_contract.py:83`, `test_benchmark.py:152`); SLOW tail (Pilot 943L, ffmpeg pairwise/concat, multi-commit) has zero markers — invisible to `-m`. `gates.sh` mypy scope (voyage + conftest + 3 property modules) ≠ `build.sh` scope (`mypy voyage` only) — intentional per 092 live-vs-snapshot but misreadable as drift. `.dockerignore *_cache/` misses `.hypothesis/` (name doesn't end in `_cache`).

## Rationale

51M cache + 348K hypothesis + stale coverage churn build context + `git status` noise; lying lock breaks reproducible slim installs; unmarked slow tail makes default suite time unpredictable.

## Live evidence

- `ls -la Voyage/ | grep -E "coverage|hypothesis|mypy|ruff"`; `du -sh .mypy_cache .hypothesis .ruff_cache .coverage`
- `grep -n "httpcore2\|httpx2\|tomli" requirements.lock pyproject.toml`
- `grep -n "markers\|not gpu" pyproject.toml scripts/gates.sh scripts/test.sh`

## Repro

```bash
ls -la .coverage .hypothesis .mypy_cache .ruff_cache 2>&1
grep -n "httpcore" requirements.lock; pip index versions httpx 2>&1 | head -5
grep -rn "pytest.mark" tests/ | head -n 20
```

## Fix candidates

1. Delete in-tree caches; CI guard fails if `ls .coverage .hypothesis .mypy_cache .ruff_cache` exists post-gate; add `.hypothesis/`+`.coverage`+`coverage.xml` to `.dockerignore` (see 090).
2. Fix or explain `httpcore2/httpx2`; add `tomli` freeze note for 3.10 (or drop conditional if floor is 3.12 — but `ruff target py310` + video python3.10 say floor stays).
3. Register `slow` marker (or document why SLOW stays unmarked); collapse gates/build mypy invocations behind `scripts/mypy-scope.sh` so live-vs-snapshot can't drift.
4. Gate: `gates.sh` green + `git status --porcelain` clean post-gate.

## Refs

- Issues 041 (coverage 65), 069 (dockerignore), 084 (dockerignore gaps in original numbering); `pyproject.toml:190-203`

## Progress log (2026-09-30, resolution pass)

- Premises re-verified live: `requirements.lock:25-26` still carries the
  `httpcore2==2.13.1` / `httpx2==2.13.1` freeze-record typo (no such PyPI
  2.x line); `tomli` conditional is `pyproject.toml:24`
  (`python_version < '3.11'`) and absent from the lock as the issue
  states; in-tree caches all present (`.coverage` 52K, `.hypothesis/`,
  `.mypy_cache/` 3.10+3.12, `.ruff_cache/`) — all gitignored;
  `.dockerignore` already lists `.hypothesis/` + `.coverage` +
  `coverage.xml` explicitly (issue 069 landed); markers remain
  `gpu` + `endurance` only, slow tail unmarked; gates/build mypy-scope
  split confirmed (`gates.sh:28` tests-inclusive vs `build.sh:16`
  `mypy voyage` only, intentional per 092).
- No code change made: every fix candidate writes outside this pass's
  scope — lockfile edits forbidden by the issue itself ("do NOT touch
  the lockfile", rows fold into 068), markers need `pyproject.toml`
  (frozen for this pass), the mypy-scope collapse needs `scripts/`
  (frozen), cache deletion touches gitignored root paths (outside
  `tests/` + `docs/TASK.md`), and registering a `slow` marker without
  the `pyproject` half would break `--strict-markers`.

## Resolution

- Not resolved here — returned as residual with premises confirmed
  current. Suggested split for scoped passes: (a) lockfile typo +
  tomli-freeze note → lock-owning track with 068; (b) `slow` marker
  registration + SLOW-tail marking → pyproject-owning track; (c)
  `scripts/mypy-scope.sh` collapse → scripts-owning track; (d) cache
  hygiene (post-gate `git status --porcelain` guard) → gates-owning
  track. None of (a)–(d) is actionable from `tests/` alone.
