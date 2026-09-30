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

## Progress log (2026-09-30, Group D pass)

- Owned follow-up (b, first half) landed: `slow` marker registered in
  `pyproject.toml` (`slow: load-sensitive or multi-second tests`, with a
  comment pinning tail-marking as residual). Additive one-liner, disjoint
  from the batch-7 coverage/gates hunks — no suite behavior changes until
  a test actually carries the marker.
- Rest re-verified, still outside scope: lockfile typo untouched
  (forbidden); tomli note untouched; gates/build mypy-scope split
  re-confirmed intentional per 092 (scripts frozen); caches still
  gitignored-present (root paths, outside scope); `.dockerignore`
  coverage re-confirmed (069 landed: `.hypothesis/` + `.coverage` +
  `coverage.xml` explicit).

## Resolution (2026-09-30, Group D pass)

- Partially resolved: `slow` marker registered; tail marking stays open.
- Files changed: `pyproject.toml` (markers list + comment only). Gate
  evidence: `--markers` lists `slow`; `-m "not slow"` selects the full
  set; qualification file 14/14 in-container. DESIGN proposals: none.
  Residuals: (a) lock typo + tomli note → lock track w/068; (b) SLOW-tail
  marking (Pilot suites — `test_tui_app.py` alone is 33 tests/34s — ffmpeg
  pairwise/concat incl. dirty `test_media_robustness_rank2.py`,
  multi-commit runs) → pyproject+tests track; (c) mypy-scope collapse →
  scripts track; (d) cache guard → gates track.

## Progress log (2026-09-30, tests-only pass)

- Premises re-verified live: `slow` marker still registered
  (`pyproject.toml:186`, comment pins tail marking as residual); zero tests
  carried `slow` before this pass (`rg pytest.mark.slow` clean); `gpu` (1)
  + `endurance` (1) unchanged; lock typo + tomli absence + cache presence
  + mypy-scope split all still hold per the Group D log — all outside
  `tests/` scope, untouched.
- Owned remainder landed (tests/ scope only): file-level `pytestmark =
  pytest.mark.slow` added to `tests/test_tui_app.py` (the cited 33-test /
  ~34 s Pilot tail — every test in the file drives `VoyageApp` via
  `run_test()` pilot, load-sensitive per the file's own 093 budgets).
  Deliberately scoped to this one file: ffmpeg pairwise/concat and
  multi-commit candidates (`test_media_robustness_rank2.py`,
  `test_generation_stack.py`, `test_integration.py`) mix fast unit tests
  with slow integration legs in the same file — file-level marking would
  over-deselect; per-test marking needs timing evidence this pass did not
  collect. No `pyproject.toml` / `scripts/` / lockfile edits (frozen).
- Gate evidence (in-container, `voyage:latest`, CPU-only):
  `--markers` lists `slow`; `tests/test_tui_app.py --collect-only` → 33
  collected; `-m "not slow"` → 33 deselected, zero collected; 2 spot tests
  pass (`test_auto_focus_targets_style_field`,
  `test_style_field_has_focus_on_mount`); `ruff check` + `ruff format
  --check` clean on the touched file. Full gates left to orchestrator
  (concurrent `supervisor.py`/`config.py` hunks hot).

## Resolution (2026-09-30, tests-only pass)

- Verdict: partial (one unambiguous slow suite marked; remainder recorded).
  Files changed: `tests/test_tui_app.py` (`pytestmark` + docstring only).
  DESIGN proposals: none.
- Residuals (exact handoff): (a) lock typo (`requirements.lock:25-26`
  `httpcore2/httpx2`) + tomli-freeze note → lock track w/068 (forbidden
  here); (b) per-test `slow` marking for the ffmpeg/multi-commit tail —
  needs timing runs to separate slow legs from fast unit tests in
  `test_media_robustness_rank2.py` / `test_generation_stack.py` /
  `test_integration.py` / `test_finalize_fastpath.py` (pyproject+tests
  track with timing evidence); (c) `scripts/mypy-scope.sh` collapse →
  scripts track; (d) post-gate cache guard → gates track. None of (a)–(d)
  is actionable from `tests/` alone beyond what landed here.

## Progress log (2026-09-30, batch 12)

- Premises re-verified live: `slow` marker still registered; 33
  `test_tui_app.py` tests still file-marked; lock typo + tomli absence
  untouched (forbidden — lock track owns `requirements.lock`).
- (a) Per-test slow marking LANDED with timing evidence (in-container
  `voyage:latest`, CPU-only, `-p no:cacheprovider --durations`):
  `test_media_robustness_rank2.py` 18 tests in 5.93s (slowest 1.24s
  publish / 1.21s staging; 12 tests <0.005s — mixed, so per-test);
  `test_generation_stack.py` 13 tests in 33.95s (8.41/8.29/8.19s
  finalize trio + 4.70s generate fallback; 8 tests <0.005s — mixed);
  `test_integration.py` 6 tests in 25.75s (15.56s + 8.47s finalize
  pair; 3 tests <1s — mixed); `test_finalize_fastpath.py` 6 tests in
  7.00s (slowest 1.99s — measured, below the marking threshold,
  left unmarked). Marked 8 tests with `@pytest.mark.slow`
  (marker lines only): media_robustness 2 (publish/staging),
  generation_stack 4 (blend/overlap/lifts/generate-fallback),
  integration 2 (finalize e2e + joint-style). File-level marking
  rejected for all four (none uniformly slow).
- (b) Cache guard → gates LANDED: `scripts/lib/common.sh` gains
  `voyage_assert_no_cache_residue` (fails loud naming leaked paths +
  the VOYAGE_CACHE_ENV contract); `scripts/gates.sh` sources it and
  calls the guard post-gate. Verified: `bash -n` clean on both;
  guard fires exit 1 on a planted `.coverage`, passes exit 0 clean.
- (c) mypy-scope collapse NOT taken: verified live in-container —
  `mypy tests/test_integration.py` reports 1 error
  (`test_integration.py:119` unused-ignore, pre-existing on a foreign
  line untouched by this pass — own diff is 2 marker lines only).
  Per the brief (collapse ONLY if all green), no list edit made.

## Resolution (2026-09-30, batch 12)

- Verdict: partial (two of three actionable legs landed; collapse
  blocked on a foreign mypy error).
  Files changed: `tests/test_media_robustness_rank2.py` /
  `tests/test_generation_stack.py` / `tests/test_integration.py`
  (8 `@pytest.mark.slow` lines only); `scripts/lib/common.sh`
  (new `voyage_assert_no_cache_residue`); `scripts/gates.sh`
  (mypy-list line minus `test_stage_timings.py` per 088 + post-gate
  guard call).
  Gate evidence: `-m "not slow"` deselects exactly the 8 marked
  (29/37 collected); `-m slow` collects 8/37 and all 8 pass (56.67s);
  `ruff check` + `ruff format --check` clean on all 4 test files;
  `mypy strict` clean on benchmark/media_robustness/generation_stack
  (integration carries the 1 foreign unused-ignore); `bash -n` clean
  on both scripts.
  DESIGN proposals: none.
- Residuals (exact handoff): (a) lock typo (`requirements.lock:25-26`)
  → lock track w/068 (forbidden here); `test_finalize_fastpath.py`
  4 ffmpeg legs (1.27–1.99s, measured, below threshold — mark when the
  slow budget tightens); `test_media_robustness_rank2.py` 4 validate
  legs (0.85–0.97s, measured, borderline); (c) mypy-scope collapse →
  scripts track once `test_integration.py:119` unused-ignore is fixed
  by its owner (remove the stale ignore or retype the
  `FinalizeOptions` call — NOT this pass: `media.py` is foreign).
