# 088 — Tests 87→~65: shared fixtures + fold clusters (same ~989 assertions)

- Severity: MEDIUM (tests / structure)
- Files: `tests/` — 87 `test_*.py`, ~989 `def test_` (2026-09-30 count; sweep said 86/972 — growth, same finding)
- Area: tests — consolidation
- Decision: Q&A locked — fold all (fixtures + folds + hygiene)

## Description

`_init_run` ×13: `conftest.py` ships `initialize_run_directory`/`run_directory_factory` (fresh `tmp_path` per call), yet `rg -c _init_run` hits commit_hardening:15, state_integrity:14, failure_policy:9, crash_matrix:8, benchmark:8, commit_split:7, generation_stack:7, … Only console + av_alignment converted. Adapter triple (`test_adapter_contract` 11 vs `test_backends_adapter` 22 vs `test_commit_split` 7) repeats fake-transport per file. Augment quad (config 31 + plan 11 + models 13 + runner 39 = 94 tests for one floor). Audio quintet (planner 13 + accounting 6 + workers 8 + request_validation 11 + take_ahead_guard 7). TUI trio (`test_tui` 20 + `test_tui_state` 26 + `test_tui_app` 33/943L Pilot, load-flaky). Video-worker quartet repeats `sys.modules` stub patterns 4×. Issue-quintet (`test_issue_014/027/029/030/032`) pins 5 perf fixes adjacently. Finalize/commit cluster: 7 files touch `validate_run`/commit/finalize. Singletons ripe for folding: `test_stage_timings` (1) → benchmark; `test_sfx_parser_parity` (2) → augment_config pattern; `test_prefetch_summary` (5) → generation_stack; `test_unset` (5) + `test_generate_blocks_request` (7) → `test_wire_contract`; `test_phase2` (3) → recovery; `test_hashing` (8) + `test_paths` (8) → `test_foundation`/`test_unit`.

## Rationale

Same assertions, ~15 fewer files, faster collection, one Pilot file, one stub helper, one fixture path. Layout-change cost (040's own argument: 20 matching edits) falls to one.

## Live evidence

- `rg -n "_init_run" tests/*.py | wc -l` → ~121 refs, 20+ files (040 re-verified)
- `wc -l tests/test_tui_app.py tests/test_causvid_worker.py tests/test_commit_hardening.py` → 943/748/495
- `ls tests/test_augment_*.py tests/test_audio_*.py tests/test_tui*.py tests/test_issue_*.py`

## Repro

```bash
rg -n "_init_run" tests/*.py | wc -l; rg -l "_init_run" tests/*.py
wc -l tests/*.py | sort -rn | head -n 15
```

## Fix candidates

1. Convert remaining `_init_run` call sites to `conftest.py` fixtures (2-3 files per pass) + `rg` ratchet that can only fall; extract shared fake-transport fixture for adapter triple; extract `fake_torch_session` helper for video-worker quartet.
2. Fold singletons + augment quad (keep runner separate — only ffmpeg/torch surface) + audio validators into `test_audio_validation.py` + TUI demotion (tui_app = only Pilot) + issue-quintet → `test_perf_regressions.py` (IDs in docstrings) + finalize merges (`final_blend_scale`→`finalize_fastpath`, `av_alignment_consumer`→`state_integrity`).
3. Trajectory 87→~65 files, same ~989 tests. Gate: `gates.sh` green, collection time recorded before/after.

## Refs

- Issues 034 (lint ratchet), 040 (`_init_run`), 041 (coverage); `tests/conftest.py:80-129`

## Progress log (2026-09-30, resolution pass)

- As-read counts drifted: **116** `test_*.py` files, **1283** `def test_`
  (issue said 87/~989 — a week of concurrent tracks). A full 116→65
  fold in one pass with 5+ agents adding test files is unsafe, so this
  pass executed the first proof fold plus the fixture convergence it
  depends on.
- Proof fold landed: `tests/test_phase2.py` (3 tests: scene-cut prefix,
  tape lookup, restart hook) moved verbatim into
  `tests/test_recovery.py` under an 088-fold header and the file
  deleted — same 3 assertions, one fewer file, recovery owns the whole
  restart/resume surface (the moved `_tape_run` already used recovery's
  helper). Safety: file re-read live, mtime + `git log -3` both
  2026-09-23, zero importers (`rg test_phase2` clean after).
- Fixture convergence (shared with 040): wrappers inlined in
  stage-timings / concept-integrity / av-alignment; new
  `tests/test_init_run_ratchet.py` guards delegation + count (144/21
  as-left). Untouched by design: the two fresh untracked files from
  concurrent agents (`test_ltxv_stage_ms`, `test_stage_a_telemetry`) and
  every `scripts/` + `voyage/` file showing in `git status`.
- As-left: 116 files (fold −1, ratchet +1), 1285 fns (ratchet +2).
- Evidence in-container: `test_recovery.py` (7 tests incl. the 3 moved)
  + neighbors → 23 passed; ruff + format + mypy green.

## Resolution

- Partially resolved (trajectory proven, not completed). Residual: the
  remaining folds (adapter triple, augment quad, audio quintet, TUI
  trio, video-worker quartet, issue-quintet, finalize/commit cluster,
  singletons) stay open — continue one cluster per pass with the same
  discipline (re-read live, check mtime/`git log`, keep assertion
  counts identical, record before/after collection time).

## Progress log (2026-09-30, Group D pass)

- Folded the issue-quintet into `tests/test_perf_regressions.py` (new):
  014 (6 tests) + 027 (9) + 029 (6) + 030 (11) + 032 (13) = **45 tests,
  same assertions, test fn names unchanged**. Cluster helpers prefixed
  `_014_`/`_027_`/`_029_`/`_030_`/`_032_` (four `Fake*` class names —
  `_FakeTensor`, `_FakeCuda`, `_FakeTorch`, `_FakeTextEncoder` — collided
  verbatim); each cluster keeps its original module docstring as a banner
  so the issue ID stays greppable. The 5 source files are deleted; `rg`
  confirms zero importers. None of the five used `_init_run`, so the
  batch-7 ratchet is untouched.
- Mid-pass collision (concurrent agent, issue 124): `video_causvid.py`
  changed `text_encoder.to("cuda")` → `to(self._device)` between the solo
  verify and the combined run, flipping 3 device assertions to
  `["cuda:0", "cpu"]`. Updated in the merged file with a 124 comment —
  intent (single shuttle: up once, park once) preserved and strengthened.
- As-left: 139 `test_*.py` files, 1486 `def test_` (fold net-zero at 45
  preserved; growth vs the 128/1366 as-read is 6+ concurrent-agent files
  and their tests landing mid-pass, plus the 4 new 132 tests).

## Resolution (2026-09-30, Group D pass)

- Partially resolved (second proof fold landed). The 150 ignore fixes were
  applied in the merged file (single site — see 150 log) rather than the
  five originals.
- Files changed: `tests/test_perf_regressions.py` (new, ~950L), deleted
  `tests/test_issue_{014_embed_restore,027_concepts_perf,029_causvid_shuttle,030_embed_bounds,032_commit_fanout}.py`.
  Gate evidence: merged file 45/45 solo; 100/100 with
  qualification/director_models_dir/generate_ensure neighbors after the
  124 adaptation; `ruff check` + `ruff format --check` clean.
  DESIGN proposals: none. Residuals: remaining clusters per the issue
  (adapter triple, augment quad, audio validators, TUI trio,
  video-worker quartet, finalize/commit merges, singletons) — one per
  pass; `gates.sh` mypy-list entries must move with any fold that deletes
  a listed file (scripts-owned, not this pass).
