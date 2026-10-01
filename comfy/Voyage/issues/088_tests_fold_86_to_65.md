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

## Progress log (2026-09-30, tests-only pass)

- Premise re-verified live: 145 `test_*.py` files (growth vs 139 as-left —
  concurrent agents still adding files); singletons `test_unset.py` (5
  tests, mtime 2026-09-29) + `test_generate_blocks_request.py` (7 tests,
  mtime 2026-09-29) both present, no helper collisions (`_multi` only in
  the request file, zero `Fake*` overlap), zero importers outside self
  (`rg from tests.test_unset|test_generate_blocks_request` clean), both in
  the `gates.sh` mypy list (80 entries). `scripts/gates.sh` itself is
  clean at runtime (`git diff --name-only -- scripts/gates.sh` empty) but
  frozen per this pass's scope contract — list-move recorded as residual.
- Fold landed per the issue's explicit singleton recipe (`test_unset` +
  `test_generate_blocks_request` → `test_wire_contract`): new
  `tests/test_wire_contract.py` (12 tests, fn names/bodies identical, each
  cluster's original module docstring kept as a banner per the batch-8
  quintet precedent); the 2 source files deleted. As-left: 144 files
  (fold −2, new +1 = net −1), 1513 `def test_` total (fold net-zero at 12
  preserved).
- Gate evidence (in-container, `voyage:latest`, CPU-only): merged + sources
  together 24/24 pre-delete; merged solo 12/12 post-delete
  (`./scripts/test.sh -q tests/test_wire_contract.py`); `ruff check` +
  `ruff format --check` clean on the new file; `mypy
  tests/test_wire_contract.py` clean (standalone — not yet in the gates
  list, see residual).

## Resolution (2026-09-30, tests-only pass)

- Verdict: fixed (one mechanical cluster folded, trajectory continues).
  Files changed: `tests/test_wire_contract.py` (new), deleted
  `tests/test_unset.py`, `tests/test_generate_blocks_request.py`.
  DESIGN proposals: none.
- Residuals (exact handoff, scripts owner): `scripts/gates.sh` mypy list
  still references the 2 deleted paths and omits the new one — until moved,
  the full `gates.sh` mypy invocation fails fast with `mypy: error:
  Cannot read file 'tests/test_unset.py'` (verified live). Move: delete
  `tests/test_unset.py` + `tests/test_generate_blocks_request.py` entries,
  add `tests/test_wire_contract.py` (alphabetical slot between
  `test_vision_metrics.py` and `test_vocoder_allowlist.py` per current
  ordering). Remaining clusters per the issue (adapter, augment quad,
  audio validators, TUI trio, video-worker quartet, finalize/commit
  merges, leftover singletons) stay open — one per pass.

## Progress log (2026-09-30, batch 12)

- Premise re-verified live: 167 `test_*.py` files (growth vs 144 as-left
  — concurrent agents still adding files); singleton
  `test_stage_timings.py` (1 test, mtime 2026-09-30, last commit
  b5cc82f batch 7) present, no helper collisions (`_committed_events`
  only in the source file; `_gauge_events`/`_init_run` only in the
  target — verified via rg), zero importers outside self (`rg
  test_stage_timings` clean except a comment ref in
  `test_stage_a_telemetry.py:10` + the gates.sh mypy entry), target
  `test_benchmark.py` imports are a superset (json/Path/
  initialize_run_directory/paths/load_config/read_state/Supervisor all
  present — no import churn needed).
- Fold landed per the issue's explicit singleton recipe
  (`test_stage_timings` → `test_benchmark`, both timing-area):
  `_committed_events` helper + 1 test moved verbatim (fn name/body
  identical, original module docstring kept as a banner per the
  batch-8 quintet precedent); the source file deleted. As-left: 166
  files (fold −1, new +1 = net 0 with the 2 split suites), test count
  net-zero at 1 preserved. `scripts/gates.sh` mypy entry removed in
  the same edit (batch-9 lesson — verified `bash -n` clean); new
  split suites stay OUT of the list per the "untracked until
  committed" rule (verified: prior split suites
  `test_registry_split`/`test_supervisor_*` are also unlisted).

## Resolution (2026-09-30, batch 12)

- Verdict: fixed (one mechanical cluster folded, trajectory continues).
  Files changed: `tests/test_benchmark.py` (+44L fold banner/helper/
  test), deleted `tests/test_stage_timings.py`,
  `scripts/gates.sh` (mypy-list line: `test_stage_timings.py` entry
  removed).
  Gate evidence (in-container, `voyage:latest`, CPU-only): merged
  solo 10/10 post-delete (`test_benchmark.py`: 9 original + 1 moved);
  `ruff check` + `ruff format --check` + `mypy strict` clean on the
  merged file; `bash -n` clean on `gates.sh`.
  DESIGN proposals: none.
- Residuals: remaining clusters per the issue (adapter triple, augment
  quad, audio validators, TUI trio, video-worker quartet,
  finalize/commit merges, leftover singletons incl.
  `test_sfx_parser_parity` 2 tests) — one per pass with the same
  discipline (re-read live, check mtime/`git log`, keep assertion
  counts identical, move the gates.sh mypy entry with any fold that
  deletes a listed file).

## Progress log (2026-09-30, batch 13)

- Premise re-verified live: 170 `test_*.py` files at pass start
  (growth vs 166 as-left — concurrent agents still adding files);
  singleton `test_sfx_parser_parity.py` (2 tests, mtime 2026-09-29,
  last commit d7b4086) present — the smallest remaining cluster per
  the issue's singleton recipe (`test_sfx_parser_parity` 2 tests →
  `test_augment_config` pattern). No helper collisions (`_sfx_defaults`
  only in the source; `_augment_defaults`/`_parse`/`_base_config` only
  in the target — verified via rg), zero importers outside self (`rg
  test_sfx_parser_parity` clean except the target's docstring mention
  + the gates.sh mypy entry), both files in the `gates.sh` mypy list.
  No longlive2 lines in either file (verified via grep) — the 079
  delete surface is untouched.
- Fold landed: `_sfx_defaults` helper + 2 tests moved verbatim (fn
  names/bodies identical, original module docstring kept as a banner
  per the batch-8 quintet precedent) into `tests/test_augment_config.py`
  (target imports already a superset: `argparse` + `build_parser`
  present — no import churn); the source file deleted.
  `scripts/gates.sh` mypy entry removed in the same edit (batch-12
  lesson — verified `bash -n` clean). As-left: 172 files (fold −1
  plus 3 concurrent-agent untracked files landing mid-pass:
  `test_issue_166_resolve_weights.py`,
  `test_longlive2_removed_079.py`, `test_registry_realesrgan_split.py`),
  test count net-zero at 2 preserved.

## Resolution (2026-09-30, batch 13)

- Verdict: fixed (one mechanical cluster folded, trajectory continues).
  Files changed: `tests/test_augment_config.py` (+31L fold banner/helper/
  tests), deleted `tests/test_sfx_parser_parity.py`,
  `scripts/gates.sh` (mypy-list line: `test_sfx_parser_parity.py` entry
  removed).
  Gate evidence (in-container, `voyage:latest`, CPU-only): merged
  solo 49/49 post-delete (`test_augment_config.py`: 31 defs incl.
  parametrized expansions + 2 moved); `ruff check` + `ruff format
  --check` + `mypy strict` clean on the merged file; `bash -n` clean
  on `gates.sh`; neighbors `test_sfx_contract` + `test_sfx_finalize` +
  `test_augment_plan` + `test_augment_models` 55 passed.
  DESIGN proposals: none.
- Residuals: remaining clusters per the issue (adapter triple, augment
  quad remainder, audio validators, TUI trio, video-worker quartet,
  finalize/commit merges, leftover singletons incl. `test_prefetch_summary`
  5 tests, `test_hashing` 8 + `test_paths` 8) — one per pass with the same
  discipline.

## Progress log (2026-09-30, batch 13 — post-pass collision note)

- After the fold landed green (49/49, ruff + format + mypy-strict
  clean), a concurrent agent's uncommitted `voyage/*` hunk broke the
  import chain: `voyage/cli_run_ops.py:28` imports
  `warn_if_deprecated_backend` from `voyage/config.py`, which their
  dirty `config.py`/`cli.py` hunk (079/longlive2 surface: untracked
  `test_longlive2_removed_079.py` present) no longer provides — so
  `tests/test_augment_config.py` (imports `voyage.cli`) now fails at
  collection with `ImportError` in-container. Foreign-caused, not this
  pass: own diff touches only `tests/` + the gates.sh list line, never
  `voyage/*`; `py_compile` clean on both touched test files (host +
  container); `ruff check` + `ruff format --check` still clean;
  `tests/test_integration.py` (no cli import chain) still 4/4 fast
  green post-hunk. No voyage/* edit made (forbidden) and no foreign
  hunk undone — the 49/49 evidence above stands as this fold's gate,
  re-run clean once their surface lands.
