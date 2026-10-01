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

## Progress log (2026-09-30, batch 13)

- (089 leg) Ad-hoc mypy run first (in-container, `voyage:latest`,
  CPU-only, no host pip): `mypy --strict tests/test_integration.py`
  → `test_integration.py:119: error: Unused "type: ignore" comment
  [unused-ignore]` (1 error, same line as batch 12 — still unused,
  condition met). Removed the stale ignore on line 119 ONLY
  (`options = FinalizeOptions(joint_style=joint_style)`); line 127
  (`FinalizeOptions(joint_style="crossfade-everything")`) still
  carries its ignore (mypy reports no unused-ignore there — it covers
  a real arg-type error, left intact).
- Fastpath legs re-timed (in-container, `--durations=10`):
  `test_finalize_fastpath.py` 6 tests in 7.35s — 4 ffmpeg legs
  1.37–2.08s (was 1.27–1.99s batch 12), 2 fast legs <0.3s (mixed
  file, so file-level marking rejected; per-test marking left to the
  slow-budget decision — still below threshold, unmarked).
- Lock legs re-verified (host grep, no edit — forbidden):
  `requirements.lock:25-26` still `httpcore2==2.13.1` /
  `httpx2==2.13.1`; `tomli` conditional still `pyproject.toml:24`
  and absent from the lock. Lock track w/068 owns both.

## Resolution (2026-09-30, batch 13)

- Verdict: partial (stale-ignore leg landed; fastpath/lock record-only).
  Files changed: `tests/test_integration.py` (1 ignore comment
  removed, line 119 only).
  Gate evidence (in-container, `voyage:latest`, CPU-only): `mypy
  --strict tests/test_integration.py` clean post-edit (no errors);
  `ruff check` + `ruff format --check` clean; `pytest
  tests/test_integration.py` 6/6 (4 fast in 1.36s + 2 slow
  16.19s/8.70s finalize pair in 24.96s).
  DESIGN proposals: none.
- Residuals (exact handoff): (a) lock typo + tomli-freeze note → lock
  track w/068 (forbidden here); (b) per-test `slow` marking for the
  4 fastpath ffmpeg legs (1.37–2.08s as-read) + 4 validate legs in
  `test_media_robustness_rank2.py` (not re-timed this pass) — mark
  when the slow budget tightens; (c) mypy-scope collapse → scripts
  track (the blocking unused-ignore is now gone — collapse is
  unblocked whenever its owner takes it).

## Progress log (2026-10-01, this pass — fastpath + validate legs marked)

- Re-timed both residual sets live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip, `-p no:cacheprovider --durations`):
  - `tests/test_finalize_fastpath.py`: 6 passed in 10.76s — 4 ffmpeg
    legs `test_finalize_native_geometry_validates_without_reencode`
    3.18s / `test_fastpath_skips_part_reencodes_but_keeps_audio` 3.15s
    / `test_build_final_audio_slice_cache_wired` 2.10s /
    `test_segment_video_matches_native_fake_segments` 1.97s (was
    1.37–2.08s batch 13, 1.27–1.99s batch 12 — all four now firmly
    multi-second); 2 fast legs (`test_slice_cache_key...` <0.005s,
    `test_cached_slice_take...` 0.21s) — file still mixed, so
    file-level marking stays rejected per the prior pass.
  - `tests/test_media_robustness_rank2.py`: 18 passed in 9.64s —
    4 validate legs `test_validate_dir_as_video...` 1.54s /
    `test_validate_dir_as_sha256...` 1.41s /
    `test_validate_deeply_nested_metrics...` 1.39s /
    `test_validate_dir_as_metrics...` 1.32s (was 0.85–0.97s batch 12
    — now at the same level as the already-slow publish leg 1.54s in
    the same file); 11 legs <0.005s + embed legs 0.01s — file still
    mixed, file-level stays rejected.
- Marked all 8 per-test `@pytest.mark.slow` (marker lines only, 4+4):
  the fastpath 4 above + the 4 validate legs. Left unmarked:
  `test_slice_cache_key...` (<0.005s pure unit) +
  `test_cached_slice_take...` (0.21s ffmpeg sine, below the
  multi-second bar) + all probe/run_capture/embed unit legs (<0.02s).
- Lock legs record-only (forbidden — lock track owns
  `requirements.lock` w/068): `requirements.lock:25-26` still
  `httpcore2==2.13.1` / `httpx2==2.13.1`; `tomli` conditional still
  `pyproject.toml:24` and absent from the lock. Untouched.
- Gates (in-container `voyage:latest`, CPU-only): `ruff check` +
  `ruff format --check` + `mypy --strict` clean on both touched
  files; full two-file run 24 passed in 17.77s; `-m slow` collects
  10/24 (4 fastpath + 6 media_robustness incl. the 2 pre-existing
  staging/publish legs) and all 10 pass in 18.59s; `-m "not slow"`
  collects 14/24.
- Concurrent-work note: the tree carries other agents' in-flight
  hunks (`pyproject.toml` 1-line, `supervisor.py`,
  `registry_records.py`, new `registry_audio`/`registry_ltxv`/
  `supervisor_tape` modules + 3 new test files, `LTX2.md`,
  `tango/Tango`) — own diff is exactly the 8 marker lines
  (verified `git diff`), nothing else touched.

## Resolution (2026-10-01, this pass)

- Verdict: fastpath + validate legs DONE (all 8 residual legs marked
  with wall-time evidence). Files changed:
  `tests/test_finalize_fastpath.py` (4 marker lines),
  `tests/test_media_robustness_rank2.py` (4 marker lines).
  DESIGN proposals: none. Test/gate evidence as in the log above.
- Residuals: (a) lock typo (`requirements.lock:25-26`) + tomli-freeze
  note → lock track w/068 (forbidden here, unchanged); (b)
  mypy-scope collapse → scripts track (batch 13 removed the blocking
  stale ignore; `scripts/` foreign here); (c) cache guard already
  landed (batch 12, gates track). Slow-budget follow-up:
  `test_cached_slice_take...` (0.21s) stays unmarked until the budget
  tightens below the multi-second bar.

## Progress log (2026-10-01, this pass — lock record + mypy-list + 8 slow marks)

- (a) Lock record-only (forbidden — never edited): `requirements.lock:25-26`
  still `httpcore2==2.13.1` / `httpx2==2.13.1`; `tomli` conditional still
  `pyproject.toml:24` and absent from the lock. Lock track w/068 owns both.
- (b) mypy-list move ATTEMPTED with per-file proof (scripts/ ownership
  granted by the brief): `tests/test_integration.py` is `mypy --strict`
  clean in-container (`Success: no issues found in 1 source file`), and
  the batch-13 blocker (line-119 stale ignore) is gone — line 119 is now
  plain `options = FinalizeOptions(joint_style=joint_style)`, line 127
  keeps its covering ignore (real arg-type error, left intact). Added
  `tests/test_integration.py` to `scripts/gates.sh:49` (alphabetical slot
  after `test_inspector_wiring.py`). Per-file gates: `mypy --strict`
  clean on `test_integration.py` + `test_state_integrity.py` +
  `test_tui.py` (3 files, no issues); `ruff check` + `ruff format
  --check` clean on all 7 touched test files; `bash -n scripts/gates.sh`
  clean. Foreign-hunk note: `scripts/gates.sh` also carries a concurrent
  hunk removing `tests/test_prefetch_shutdown.py` from the same mypy
  list (not mine — preserved, never touched); full `gates.sh` green is
  left to the orchestrator because the file is shared.
- (c) Slow-tail re-timed live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip, `-m "not slow and not gpu and not endurance"`
  `--durations=30`, 1657 passed in 257.07s): top unmarked multi-second
  legs are `test_e1_media_augment.py::test_188_numbering_gap_lenient...`
  9.98s / `test_generate.py::test_generate_fake_end_to_end...` 9.32s /
  `test_media_memory.py::test_finalize_reencode...` 8.97s /
  `test_state_integrity.py::test_finalize_skip_bad...` 6.47s /
  `test_e1_media_augment.py::test_138_missing_video_lenient...` 5.64s /
  `test_tui.py::test_generate_end_to_end_fake_backend` 5.64s /
  `test_e1_media_augment.py::test_138_missing_audio_lenient...` 5.50s /
  `test_failure_policy.py::test_finalize_space_preflight` 5.09s. Marked
  all 8 per-test `@pytest.mark.slow` (marker lines only, 6 files).
  Verified: `-m slow --collect-only` on the 7 touched files collects
  10 (8 new + 2 pre-existing integration); 5 passed in 28.83s
  (e1 3 + failure 1 + state 1) and 3 passed in 23.04s (generate +
  media_memory + tui). Remaining 22 unmarked multi-second legs recorded
  below with the same evidence run (all `not slow` at run time).

## Resolution (2026-10-01, this pass)

- Verdict: partial (lock record-only; mypy-list landed per-file-green
  with a shared-file caveat; 8 slow legs marked with wall-time evidence).
  Files changed: `scripts/gates.sh` (1 mypy-list token,
  `test_integration.py`); `tests/test_e1_media_augment.py` (3 markers),
  `tests/test_generate.py` (1), `tests/test_media_memory.py` (1),
  `tests/test_state_integrity.py` (1), `tests/test_tui.py` (1),
  `tests/test_failure_policy.py` (1); this issue file.
  DESIGN proposals: none.
- Residuals: (a) lock typo (`requirements.lock:25-26`) + tomli note →
  lock track w/068 (forbidden, unchanged); (b) full `gates.sh` green →
  orchestrator (shared file carries the foreign prefetch-shutdown hunk);
  comment staleness (`scripts/gates.sh:27-28` still says 83 files vs 76
  listed) left as-is (list-line only per the brief); (c) 22 unmarked
  multi-second legs from the same 257s evidence run (mark when the slow
  budget tightens): `test_generate.py` 4.75/4.72/4.65s (name-wins,
  ensure-selective, defaults, name-routes), `test_generate_ensure.py`
  4.74/4.73s, `test_av_alignment_consumer.py` 3.53s,
  `test_media_memory.py::test_finalize_fastpath...` 3.00s,
  `test_ltxv.py::test_verify...` 2.66s,
  `test_tui_checkbox_toggle_181.py` 2.49s, `test_scoreboard.py` 2.43s,
  `test_cli_validate_handoff.py` 2.40s,
  `test_commit_side_integrity_095_101_104.py` 2.27/2.02s,
  `test_e1_media_augment.py::test_188_strict...` 2.13s,
  `test_qualification.py::test_fake_three_segment...` 2.07s,
  `test_generation_stack.py::test_stop_workers...` 2.01s,
  `test_console.py` 1.99s, `test_issue_141...` 1.97s,
  `test_supervisor_av_align.py` 1.94/1.85s,
  `test_e1_media_augment.py::test_138_strict...` 1.90s,
  `test_crash_matrix.py::test_repeated_crashes...` 1.78s.

## Progress log (2026-10-01, record-only maintenance pass — lock leg only)

- Lock leg record-only (lockfile FORBIDDEN — never edited, per brief):
  `requirements.lock:25-26` still `httpcore2==2.13.1` /
  `httpx2==2.13.1` (host `sed` read); `tomli` conditional still
  `pyproject.toml:24` (`python_version < '3.11'`) and absent from the
  lock (host `grep` clean). `slow` marker still registered
  (`pyproject.toml:185`). Lock track w/068 owns both — untouched.
- No other 089 leg attempted (slow-tail, mypy-list, cache-guard legs
  belong to their owning tracks; this pass is the lock record only).
  No test run for this leg (record-only, no behavior change).

## Resolution (2026-10-01, record-only maintenance pass — lock leg)

- Verdict: DEFERRED (record-only) — lock premises hold, file untouched.
  Files changed: none for 089 (this issue file only).
  Gate evidence: n/a (no change). DESIGN proposals: none.
- Residuals: unchanged — (a) lock typo (`requirements.lock:25-26`) +
  tomli-freeze note → lock track w/068 (forbidden here).

## Progress log (2026-10-01, lock-leg re-verdict — user-ordered, read-first)

- Batch-2's 068 refutation re-verified LIVE (in-container
  `voyage:latest`, CPU-only, no host pip): `pip show
  httpx2/httpcore2/huggingface_hub` -> 2.13.1/2.13.1/2.0.0 with intact
  `Required-by` chains (`httpx2 <- huggingface_hub`,
  `httpcore2 <- httpx2`); `importlib.metadata` Requires-Dist:
  `huggingface_hub==2.0.0` declares `httpx2<3,>=2.0.0`
  unconditional/marker-free and `httpx2==2.13.1` declares
  `httpcore2==2.13.1` pinned; all three import with versions.
  Gate `tests/test_lock_manifest_agreement.py:172` pins both rows
  reachable (green in this pass's 31-test neighbor run).
- tomli note checked likewise: `pyproject.toml:24` conditional
  `tomli>=2.0; python_version < '3.11'`; `requirements.lock` is the
  slim-image 3.12 freeze (header declares slim-only scope) and
  correctly omits it — absence is marker-correct, not a defect.
  Runtime uses stdlib `tomllib` with `tomli` fallback
  (`voyage/config.py:18-20`, same in `voyage/tui_state.py`); the 3.10
  worker image carries its own stack (`huggingface_hub[cli]==0.36.2`
  layer, no lock coverage by design).
- Verdict: MOOT — neither lock leg describes a real defect anymore.
  The `httpcore2/httpx2` rows are genuine distributions (068 stands,
  re-proven live); the tomli absence is conditional-correct.
  CLOSE-recommendation for the lock legs; any worker-path freeze note,
  if wanted, belongs to the lock/worker track as a nicety, never as a
  defect. No source edits (record-only per brief).

## Resolution (2026-10-01, lock-leg re-verdict)

- Verdict: lock legs MOOT / CLOSE-recommended with live evidence
  above. Files changed: this issue file only (append). Gate
  evidence: `test_lock_manifest_agreement` green; no behavior
  change. DESIGN proposals: none.
- Residuals: slow-tail / mypy-scope / cache-guard legs per their
  owning tracks (unchanged by this pass).
