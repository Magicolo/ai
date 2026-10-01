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

## Progress log (2026-10-01, prefetch-summary fold pass)

- Premise re-verified live: 166 `test_*.py` files at pass start;
  singleton `test_prefetch_summary.py` (5 tests, mtime 2026-09-25, last
  commit d7b4086) present — the smallest remaining cluster per the
  issue's singleton recipe (`test_prefetch_summary` 5 tests →
  `test_generation_stack`). No helper collisions (`_events` only in the
  source; `_metric_events`/`_write_toml`/`_init_run`/`_commit_two` only
  in the target — verified via rg), zero importers outside self (`rg
  test_prefetch_summary` clean except the gates.sh mypy entry + issue
  docs + the new fold banner), both files in the `gates.sh` mypy list.
  No longlive2 lines in either file — the 079 delete surface is
  untouched.
- Fold landed: `_events` helper + 5 tests moved verbatim (fn
  names/bodies identical, original module docstring kept as a banner
  per the batch-8 quintet precedent) into
  `tests/test_generation_stack.py` (target imports extended minimally:
  `from typing import Any` + `summarize_prefetch_outcome` alongside the
  existing `Supervisor` import — no other churn); the source file
  deleted. `scripts/gates.sh` mypy entry removed in the same edit
  (batch-12 lesson — verified `bash -n` clean; target entry already
  present, so only a deletion). As-left: 168 files (fold −1 plus the
  new `test_supervisor_tape_helpers.py` from the paired 081 pass plus 2
  concurrent-agent untracked files
  `test_registry_audio_split.py`/`test_registry_ltxv_split.py` landing
  mid-pass), test count net-zero at 5 preserved (merged file 13→18).
- Gate evidence (in-container, `voyage:latest`, CPU-only): merged +
  sources together 23/23 pre-delete; merged solo 18/18 post-delete;
  neighbors `test_supervisor_prefetch_helpers` +
  `test_prefetch_shutdown` + `test_prefetch_invalidated_136_168` +
  `test_generation_stack` 29 passed; `ruff check` + `ruff format
  --check` + `mypy strict` clean on the merged file; `bash -n` clean
  on `gates.sh`. Foreign hunks in `tests/test_finalize_fastpath.py` /
  `tests/test_media_robustness_rank2.py` + many `issues/*` edits coexist
  untouched.

## Resolution (2026-10-01, prefetch-summary fold pass)

- Verdict: fixed (one mechanical cluster folded, trajectory continues).
  Files changed: `tests/test_generation_stack.py` (+56L fold banner/helper/
  tests, 13→18 tests), deleted `tests/test_prefetch_summary.py`,
  `scripts/gates.sh` (mypy-list line: `test_prefetch_summary.py` entry
  removed).
  Gate evidence (in-container, `voyage:latest`, CPU-only): merged
  solo 18/18 post-delete; `ruff check` + `ruff format --check` +
  `mypy strict` clean on the merged file; `bash -n` clean on `gates.sh`;
  neighbors 29 passed.
  DESIGN proposals: none.
- Residuals: remaining clusters per the issue (adapter triple, augment
  quad remainder, audio validators, TUI trio, video-worker quartet,
  finalize/commit merges, leftover singletons incl. `test_hashing` 7 +
  `test_paths` 8) — one per pass with the same discipline.

## Progress log (2026-10-01, prefetch-shutdown fold pass)

- Premise re-verified live: singleton `test_prefetch_shutdown.py` (2
  tests, mtime 2026-09-30, last functional commit 195b267) present —
  the smallest remaining cluster. No helper/class collisions
  (`_RecordingDirector`/`_WedgedDirector`/`_stub_decision`/
  `_stubbed_supervisor`/`_submit_prefetch`/`_shutdown_executor` absent
  from the target — verified via rg), zero importers outside self
  (`rg test_prefetch_shutdown` clean except the gates.sh mypy entry +
  issue docs), both files in the `gates.sh` mypy list.
  Target `test_generation_stack.py` already holds the prior
  prefetch-summary fold (same area — shutdown-drain belongs with the
  prefetch contract), so no new file is created.
- Fold landed: 2 stub-director classes + 4 helpers + 2 tests moved
  verbatim (fn names/bodies identical, original module docstring kept
  as a banner at `test_generation_stack.py:326` per the batch-8
  quintet precedent) into `tests/test_generation_stack.py` (target
  imports extended minimally: `math`/`threading`/`time`/
  `ThreadPoolExecutor`/`ConceptStore`/`ProjectConfig`/
  `DirectorDestination`/`EvolutionDecision`/`StyleSpec`/
  `PREFETCH_TIMEOUT_SECONDS` alongside the existing `Supervisor`
  import — no other churn); the source file deleted.
  `scripts/gates.sh` mypy entry removed in the same edit (batch-12
  lesson — verified `bash -n` clean; target entry already present,
  so only a deletion). As-left: 169 files, merged file 18→20 tests,
  test count net-zero at 2 preserved.
- Gate evidence (in-container, `voyage:latest`, CPU-only): merged +
  sources together 20/20 pre-delete; merged solo 20/20 post-delete;
  neighbors `test_supervisor_prefetch_helpers` +
  `test_supervisor_proposal_helpers` +
  `test_supervisor_commit_types` + `test_supervisor_tape_helpers` +
  `test_prefetch_invalidated_136_168` 22 passed; `ruff check` + `ruff
  format --check` clean on the merged file; `mypy strict` reports
  zero errors in the merged file (4 errors all inside the foreign
  dirty `voyage/registry_records.py` — concurrent agent's in-flight
  causvid/sfx facades, followed via imports, recorded not fixed);
  `bash -n` clean on `gates.sh`.

## Resolution (2026-10-01, prefetch-shutdown fold pass)

- Verdict: fixed (one mechanical cluster folded, trajectory continues).
  Files changed: `tests/test_generation_stack.py` (+149L fold banner/
  classes/helpers/tests, 18→20 tests), deleted
  `tests/test_prefetch_shutdown.py`, `scripts/gates.sh` (mypy-list
  line: `test_prefetch_shutdown.py` entry removed).
  Gate evidence (in-container, `voyage:latest`, CPU-only): merged
  solo 20/20 post-delete; `ruff check` + `ruff format --check`
  clean on the merged file; `mypy strict` zero errors in the merged
  file (foreign registry errors recorded, not fixed); `bash -n`
  clean on `gates.sh`; neighbors 22 passed.
  DESIGN proposals: none.
- Residuals: remaining clusters per the issue (adapter triple, augment
  quad remainder, audio validators, TUI trio, video-worker quartet,
  finalize/commit merges, leftover singletons incl. `test_hashing` 7 +
  `test_paths` 8) — one per pass with the same discipline (re-read
  live, check mtime/`git log`, keep assertion counts identical, move
  the gates.sh mypy entry with any fold that deletes a listed file).

## Progress log (2026-10-01, final-blend-scale fold pass)

- Premise re-verified live: 170 `test_*.py` files at pass start;
  `tests/test_final_blend_scale.py` (4 tests, mtime 2026-09-30,
  last functional commit 703225d) present — the smallest remaining
  cluster with an explicit singleton recipe (fix candidate 2:
  `final_blend_scale`→`finalize_fastpath`, same finalize-audio
  area). Collision check via rg/diff: `_sine_take` defined in BOTH
  files but byte-identical (verified via diff pre-fold — reused,
  not duplicated); `_synthetic_run`/`_input_count`/
  `FRAMES_PER_SEGMENT`/`FPS`/`SEGMENT_SECONDS` only in the source;
  `_init_run`/`_commit_two`/slice-cache helpers only in the target
  — no other collisions. Zero importers outside self (`rg
  test_final_blend_scale` clean except two docstring mentions in
  `test_issue_152_{blend_probe_memo,wide_manual_join_proof}.py` +
  the gates.sh mypy entry + issue docs), both files in the
  `gates.sh` mypy list.
- Fold landed: fold banner (original module docstring) +
  constants + 2 helpers + 4 tests moved verbatim (fn names/bodies
  identical, `_sine_take` dedup documented in a NOTE at the fold
  site) into `tests/test_finalize_fastpath.py` (target imports
  extended minimally: `concurrent.futures` only — `json`/
  `subprocess`/`paths`/`build_final_audio`/`probe` already present;
  `_blend_pair`/`assemble_segment_audio`/`write_segment_manifest`
  stay function-local as in the source); the source file deleted.
  `scripts/gates.sh` mypy entry removed in the same edit (batch-12
  lesson — verified `bash -n` clean; target entry already present,
  so only a deletion). As-left: 170 files (fold −1 plus concurrent
  `test_registry_director_split.py` landing mid-pass), merged file 6→10 tests,
  test count net-zero at 4 preserved.
- Gate evidence (in-container, `voyage:latest`, CPU-only): both
  files together 10/10 pre-delete (9.91s baseline); merged solo
  10/10 post-delete; neighbors 28 passed
  (`test_issue_152_blend_probe_memo` +
  `test_issue_152_wide_manual_join_proof` +
  `test_supervisor_{proposal,prefetch,commit_types,tape}_helpers`);
  `ruff check` + `ruff format --check` + `mypy strict` clean on the
  merged file; `bash -n` clean on `gates.sh`. Foreign hunks in
  `voyage/supervisor.py` / `tests/test_stage_a_telemetry.py` /
  `LTX2.md` coexist untouched (full `gates.sh` not run — they would
  color it).

## Resolution (2026-10-01, final-blend-scale fold pass)

- Verdict: fixed (one mechanical cluster folded, trajectory continues).
  Files changed: `tests/test_finalize_fastpath.py` (+147L fold banner/
  constants/helpers/tests, 6→10 tests), deleted
  `tests/test_final_blend_scale.py`, `scripts/gates.sh` (mypy-list
  line: `test_final_blend_scale.py` entry removed).
  Gate evidence (in-container, `voyage:latest`, CPU-only): merged
  solo 10/10 post-delete; `ruff check` + `ruff format --check` +
  `mypy strict` clean on the merged file; `bash -n` clean on
  `gates.sh`; neighbors 28 passed.
  DESIGN proposals: none.
- Residuals: remaining clusters per the issue (adapter triple:
  `tests/test_adapter_contract.py` 11 vs
  `tests/test_backends_adapter.py` 22 vs `tests/test_commit_split.py`
  7; augment quad remainder; audio validators; TUI trio; video-worker
  quartet; finalize/commit merges incl. `test_av_alignment_consumer.py`
  6 → `test_state_integrity.py`; leftover singletons incl.
  `tests/test_hashing.py` 7 + `tests/test_paths.py` 8 →
  `test_foundation`/`test_unit.py`) — one per pass with the same
  discipline (re-read live, check mtime/`git log`, keep assertion
  counts identical, move the gates.sh mypy entry with any fold that
  deletes a listed file).

## Progress log (2026-10-01, full-resolution pass)

- Premise: 170 `test_*.py` at pass start; user ordered FULL resolution
  of every remaining cluster in one pass under the established recipe
  (re-read live, mtime/`git log`, names/bodies/counts identical,
  banner, zero importers, gates-list move in the same edit, `bash -n`
  clean; TDD per fold: merged+sources green pre-delete, merged solo
  green post-delete; ruff + format + mypy strict per touched file; one
  cluster at a time with `git diff` disjointness vs concurrent hunks).
- Six folds landed (11 source files deleted, assertion net-zero
  throughout — every moved test fn keeps its name, every assertion
  line is byte-identical unless a drift adaptation is noted):
  1. `test_av_alignment_consumer` (6) → `test_state_integrity`
     (15→21): `_commit` byte-identical — reused, not duplicated (blend
     precedent); import extended with the three media helpers only.
     Pre-delete 27 (merged 21 + 6 dupes), post-delete 21/21.
     Gates: source entry removed (target kept).
  2. `test_hashing` (7) + `test_paths` (8) → `test_unit` (22→37):
     no helper collisions; imports extended (hashlib, hashing/media/
     paths modules, sha256 helpers, path constants, MediaError, worker
     modules). Pre-delete 52 (37 + 15 dupes), post-delete 37/37.
     Gates: both source entries removed (target kept).
  3. Adapter triple → `test_adapter_contract` (11→40): backends
     `_stub_transport` semantically identical — reused (NOTE); backends
     `_video_config`/`_request` + commit-split helpers verbatim.
     Drift adaptations (assertions untouched, 124 precedent):
     backends `[literal-required]`→`[index]`, unknown-backend
     +`[arg-type]`, pre-loop `worker_result: dict[str, Any]`
     declaration (annotated `for` target is a SyntaxError — first
     attempt proved it); commit-split per-line ignore codes exactly as
     merged-solo mypy demands (incompatible assigns `assignment`,
     compatible restores `method-assign`; state-line `arg-type` kept,
     config-line + backend-line stale ignores dropped); supervisor
     re-export keeps `attr-defined` (import source unchanged).
     Pre-delete 69 (40 + 29 dupes), post-delete 40/40. Gates: no
     change (both sources unlisted).
  4. `test_audio_take_ahead_guard` (7) → `test_audio_request_validation`
     (11→18): no collisions; imports extended (pydantic,
     AudioConfig/toml/resolve). Pre-delete 25 (18 + 7 dupes),
     post-delete 18/18. Gates: source entry removed (target kept).
  5. Four TUI satellites (14) → `test_tui` (19→33): absent-defaults
     (4) + checkbox-help (3) + checkbox-toggle (2) + progress-parity
     (5). `_require_app` ×2 byte-identical — kept once (NOTE);
     `_isolated_home` ×3 behavior-identical (docstring differs) —
     target's kept (NOTE); absent-defaults tests gain the autouse
     fixture as a no-op (NOTE, they never read HOME). Pre-delete 47
     (33 + 14 dupes), post-delete 33/33. Gates: absent-defaults entry
     removed (other three sources unlisted; target kept).
  6. `test_integration` (6) → `test_state_integrity` (21→27):
     `_init_run` behavior differs (fixed itest/seed-7 vs parameterized)
     — moved as `_itest_init_run` (quintet prefix precedent); imports
     extended (FinalizeOptions/validate_video, read_state,
     SubprocessWorker). Pre-delete 33 (27 + 6 dupes), post-delete
     27/27. Gates: source entry removed (target kept).
- Gate evidence (in-container, `voyage:latest`, CPU-only): every fold
  `ruff check` + `ruff format --check` + `mypy strict` clean on all
  touched files (final combined run: 5 files green); `bash -n` clean
  on `gates.sh` after each list edit; gates-list integrity re-verified
  (every listed path exists — no dangling entries). Full `gates.sh`
  not run (foreign dirty hunks in `voyage/*` + `tests/test_registry_split.py`
  / `tests/test_stage_a_telemetry.py` would color it — untouched per
  scope). Slow suites included post-delete: state_integrity 27/27
  (~44s), tui 33/33 (~10s), adapter 40/40, unit 37/37, audio 18/18.
- As-left: 163 `test_*.py` files, 1625 `def test_` (fold net-zero;
  file-count delta vs 170 as-read = −11 folded + 4 concurrent-agent
  adds landing mid-pass, incl. untracked
  `test_supervisor_routing_helpers.py`).
- Never touched: `000_INDEX.md`, `DESIGN.md`, `AGENTS.md`, other
  issues' files, any `voyage/*` source (only `tests/*` + the owned
  `scripts/gates.sh` list lines). Never committed. No host pip (all
  probes/tests via `docker run --rm -v $PWD:/app -w /app voyage:latest`).

## Resolution (2026-10-01, full-resolution pass)

- Verdict: fixed to the recipe limit — six folds landed (files
  changed: `tests/test_adapter_contract.py` 11→40 tests,
  `tests/test_audio_request_validation.py` 11→18,
  `tests/test_tui.py` 19→33, `tests/test_state_integrity.py` 15→27,
  `tests/test_unit.py` 22→37; deleted 11 sources:
  `test_av_alignment_consumer`, `test_hashing`, `test_paths`,
  `test_backends_adapter`, `test_commit_split`,
  `test_audio_take_ahead_guard`, `test_tui_absent_defaults_023`,
  `test_tui_checkbox_help_114`, `test_tui_checkbox_toggle_181`,
  `test_tui_progress_parity_113`, `test_integration`;
  `scripts/gates.sh` mypy list: 6 source entries removed —
  av_alignment_consumer, hashing, paths, take_ahead_guard,
  tui_absent_defaults_023, integration — each in the same edit as its
  deletion; backends_adapter/commit_split/checkbox_help/
  checkbox_toggle/progress_parity were unlisted, no move needed).
  DESIGN proposals: none.
- Residuals (exact causes — each cluster probed live, unfoldable
  verbatim under the recipe):
  - Augment quad remainder (`test_augment_config` 33 +
    `test_augment_plan` 11 + `test_augment_models` 13 +
    `test_augment_runner` 39 + `test_augment_weight_loading` 6):
    SKIPPED — foreign hunk in `tests/test_augment_models.py`
    (concurrent agent's uncommitted +2 lines adding
    `verify_ltx25_models`/`verify_ltx23_models` to the expected tuple,
    matching their untracked `voyage/registry_ltx23.py` +
    `voyage/registry_ltx25.py`; registry-split track actively working
    this area). Rule: never fold out of a file another agent just
    edited. Retry once their track lands (re-read + `git diff` first).
  - Video-worker quartet: SKIPPED — 24 pre-existing mypy-strict
    errors across the candidates (`test_causvid_worker` 14 incl.
    `video_causvid.CAUSVID_COMMIT`/`CAUSVID_CHECKPOINT_FILE`
    attr-defined; `test_video_common` 6 incl. `TAIL_FILENAME`/
    `TAPE_FILENAME` attr-defined on both worker modules + 2 ndarray
    type-arg; `test_ltxv_tensor_handoff` 4 incl. ndarray type-arg): the
    6 attr-defined errors need `voyage/workers/*` re-exports, which is
    voyage-source scope (forbidden to this pass + other groups' area).
    Transplanting them into the listed+clean `test_ltxv.py` would turn
    a green gated file red. Owners: worker/registry tracks.
  - Audio remainder (`test_audio_planner` 13 + `test_audio_accounting`
    6 + `test_audio_workers` 8 + `test_audio_acestep_cwd` 2): planner
    carries 2 pre-existing errors (ndarray type-arg/no-any-return,
    unlisted); the rest are clean but have no enumerated fold target
    left (validators done). One-per-pass follow-up.
  - TUI remainder (`test_tui_app` 33 Pilot 989L + `test_tui_state` 26):
    app stays solo per the issue's own demotion recipe (tui_app = only
    Pilot); state carries 3 pre-existing comparison-overlap errors
    (pytest.approx tuple equality, unlisted) — needs owner attention,
    not a verbatim fold into listed `test_tui.py`.
  - Finalize/commit remainder (`test_commit_hardening` 20 +
    `test_finalize_encode_rank2` 8; `test_finalize_fastpath` 10 stays
    as the area owner): hardening carries 5 pre-existing errors incl.
    `supervisor.validate_video` attr-defined (needs voyage-source
    re-export — forbidden); encode_rank2 carries 1 (`audio_acestep.
    subprocess` attr-defined — needs worker re-export or body edits).
    Untracked `test_commit_slice_compensation.py` excluded (in-flight,
    uncommitted — never fold untracked files).
  - All residuals keep assertion counts identical (nothing moved);
    next passes use the same discipline. `gates.sh` list is fully
    consistent (no dangling paths).

## Progress log (2026-10-01, tui-state fold + collapse-support pass)

- Premise re-verified live: 165 `test_*.py` at pass start (growth vs
  163 as-left — 2 concurrent-agent untracked files
  `test_issue_152_single_graph.py` /
  `test_issue_166_chunk_worker.py` landing mid-pass, left untouched
  per the untracked rule); `tests/test_augment_models.py` still
  carries the foreign uncommitted +2-line hunk
  (`verify_ltx25_models`/`verify_ltx23_models`, matching their
  untracked `voyage/registry_ltx23.py` + `voyage/registry_ltx25.py`)
  — augment quad still SKIPPED per the never-fold-foreign rule.
- Fold landed: `tests/test_tui_state.py` (26 tests) → `tests/test_tui.py`
  (33→59 defs, same TUI area as the prior satellite folds). Verified
  live first: mtime/`git log` (state 2026-09-30, tui 2026-10-01),
  zero importers outside self (`rg test_tui_state` clean except the
  target's docstring mention + gates.sh entry + issue docs), no test
  fn collisions (26 vs 33, disjoint), no helper collisions
  (`_filled_state` only in source; `_valid_state`/`_isolated_home`/
  `_customized_base` only in target), both files in the gates.sh mypy
  list (source added by the paired 089 leg this same pass, so the
  list move is a pure deletion). Moved verbatim (fn names/bodies
  identical, original module docstring kept as a banner at the fold
  site with the autouse-fixture NOTE); target imports extended
  minimally (7 missing `tui_state` names added alongside the existing
  import — no other churn); source file deleted. `scripts/gates.sh`
  mypy entry removed in the same edit (batch-12 lesson — verified
  `bash -n` clean; target entry kept).
- TDD/gates (in-container, `voyage:latest`, CPU-only, no host pip):
  pre-delete both-together 89/89 passed (33+26 defs with parametrized
  expansions); merged solo 89/89 post-delete (net-zero, same 89
  instances); `ruff check` + `ruff format --check` + `mypy strict`
  clean on the merged file; `bash -n` clean on `gates.sh`.
  As-left: 164 `test_*.py` files (fold −1 plus the 2 untracked
  concurrent adds), merged file 59 defs.
- 089-collapse support landed in the same pass (test-scope only, own
  `tests/` + owned `scripts/gates.sh` list lines): 17 error files
  fixed to per-file green (unused-ignore removals, ignore-code
  corrections `method-assign`→`assignment` / `arg-type`→`call-overload`,
  `ndarray[Any, Any]` annotations, `_base_config` typing,
  `Frame: TypeAlias`, `comparison-overlap` ignores, intentional-invalid
  `arg-type` ignores) — see 089 log for the per-file list. This
  unblocks future folds (e.g. audio planner is now clean) but creates
  no new fold target by itself.
- Never touched: `000_INDEX.md`, `DESIGN.md`, `AGENTS.md`, other
  issues' files, any `voyage/*` source (only `tests/*` + the owned
  `scripts/gates.sh` list lines), foreign dirty hunks
  (`test_augment_models.py`, `test_generate_ensure.py`,
  `test_registry_split.py`, `test_run_sh.py`,
  `test_finalize_fastpath.py`, all `voyage/*` hunks) and untracked
  files. Never committed. No host pip (all probes/tests via
  `docker run --rm -v $PWD:/app -w /app voyage:latest`).

## Resolution (2026-10-01, tui-state fold + collapse-support pass)

- Verdict: fixed (one mechanical cluster folded, trajectory continues;
  collapse-support fixes unblock follow-ups).
  Files changed: `tests/test_tui.py` (+303L fold banner/helper/
  26 tests, 33→59 defs), deleted `tests/test_tui_state.py`,
  `scripts/gates.sh` (mypy-list line: `test_tui_state.py` entry
  removed); plus the 17 test-scope mypy fixes listed in the 089 log
  (own `tests/` scope, each per-file green with pytest evidence).
  Gate evidence (in-container, `voyage:latest`, CPU-only): merged
  solo 89/89 post-delete (matches the 89/89 both-together baseline);
  `ruff check` + `ruff format --check` + `mypy strict` clean on the
  merged file; `bash -n` clean on `gates.sh`.
  DESIGN proposals: none.
- Residuals: remaining clusters per the issue (adapter triple DONE
  prior pass; augment quad remainder SKIPPED — foreign hunk in
  `test_augment_models.py`, retry once their track lands; video-worker
  quartet SKIPPED — `test_causvid_worker` 14 + `test_video_common` 6
  incl. `TAIL_FILENAME`/`TAPE_FILENAME`/`CAUSVID_COMMIT`/
  `CAUSVID_CHECKPOINT_FILE` attr-defined needing `voyage/workers/*`
  re-exports (voyage-source scope, forbidden); audio remainder —
  no same-area fold target left (validators done; planner/accounting/
  workers/acestep are distinct areas, all now mypy-clean for future
  passes); TUI remainder `test_tui_app` 33 Pilot stays solo per the
  issue's own demotion recipe (file-marked slow, load-flaky);
  finalize/commit remainder SKIPPED — `test_commit_hardening` 5 +
  `test_finalize_encode_rank2` 1 both attr-defined needing
  voyage-source re-exports) — one per pass with the same discipline
  (re-read live, check mtime/`git log`, keep assertion counts
  identical, move the gates.sh mypy entry with any fold that deletes
  a listed file).

## Progress log (2026-10-01, remainders assessment + unblock pass)

- Premise: `git diff` checked FIRST (repo root
  `/home/goulade/Projects/ai`); the paired 089 leg in this same
  pass closed all 8 mypy blocks with facade-only voyage
  re-exports (see 089 log: mypy 34→0, 138/138 pytest, 8 gates.sh
  entries). Every remaining 088 cluster re-probed live against
  the current tree (164 `test_*.py` files; counts re-read, not
  remembered):
  - Augment quad remainder (`test_augment_config` 33 +
    `test_augment_plan` 11 + `test_augment_models` 13 +
    `test_augment_runner` 39 + `test_augment_weight_loading` 6):
    STILL SKIPPED — the foreign `+2`-line hunk in
    `tests/test_augment_models.py`
    (`verify_ltx25_models`/`verify_ltx23_models`, matching their
    untracked `voyage/registry_ltx23.py` +
    `voyage/registry_ltx25.py`) is still uncommitted. Rule:
    never fold out of a file another agent just edited. Retry
    once their track lands (re-read + `git diff` first).
  - Video-worker quartet (`test_causvid_worker` 32 +
    `test_video_common` 10 + `test_ltxv` 21 +
    `test_ltxv_tensor_handoff` 11): mypy block GONE (paired 089
    leg fixed all 24 errors), but NO FOLD landed — the issue
    never enumerates a fold target for this cluster (no
    "X → Y" recipe), and `test_ltxv_tensor_handoff.py` carries
    an uncommitted hunk (the paired collapse-pass ndarray fix,
    own line but still dirty). One-per-pass follow-up with an
    explicit target.
  - Audio remainder (`test_audio_planner` 13 +
    `test_audio_accounting` 6 + `test_audio_workers` 8 +
    `test_audio_acestep_cwd` 2): NO FOLD — validators done, the
    four are distinct areas with no enumerated same-area target;
    `test_audio_planner.py` carries an uncommitted hunk
    (paired collapse-pass ndarray fix). All four are now
    mypy-clean (paired leg + collapse) for future passes.
  - TUI remainder (`test_tui_app` 33 Pilot): STAYS SOLO per the
    issue's own demotion recipe (tui_app = only Pilot;
    file-marked slow, load-flaky). Closed by design, not
    blocked.
  - Finalize/commit remainder (`test_commit_hardening` 20 +
    `test_finalize_encode_rank2` 8; `test_finalize_fastpath` 10
    stays as the area owner): SKIPPED — the owner target
    `tests/test_finalize_fastpath.py` carries an uncommitted
    foreign hunk (issue-152 single-graph join rewrite, +8/-3
    lines), and `test_commit_hardening.py` was just touched by
    the paired 089 leg (ignore-code fixes, dirty). Folding
    dirty-into-dirty violates the quiet-regions rule. Retry once
    both settle.
  - Leftover singletons: NONE — every enumerated singleton
    (`stage_timings`, `sfx_parser_parity`, `prefetch_summary`,
    `prefetch_shutdown`, `final_blend_scale`, `unset`,
    `generate_blocks_request`, `hashing`, `paths`,
    `av_alignment_consumer`, `integration`) is already folded.
    The remaining ≤3-test files on disk are concurrent-track
    split suites outside this issue's scope (never fold
    untracked/foreign files).
- No fold landed this pass (zero files moved, assertion counts
  untouched); the pass's value is the unblock (all 8 mypy legs
  green + listed, so every cluster above is now foldable on
  mypy grounds — only edit-collision and missing-target causes
  remain). Discipline kept throughout: re-read live, `git diff`
  before every edit, quiet regions only, no `voyage/*` behavior
  change, no untracked-file folds.
- Never touched: `000_INDEX.md`, `DESIGN.md`, `AGENTS.md`, other
  issues' files, foreign dirty hunks, untracked files. Never
  committed. No host pip (all probes/tests via
  `docker run --rm -v $PWD:/app -w /app voyage:latest`).

## Resolution (2026-10-01, remainders assessment + unblock pass)

- Verdict: unblocked, not folded (no mechanical cluster met the
  quiet-target recipe this pass). Files changed for 088: none
  (the paired 089 leg's 8 re-exports + 5 test fixes + 8 gates.sh
  entries are recorded in 089). Gate evidence: 089 per-file
  gates green (mypy 34→0, ruff + format clean, 138/138 pytest);
  no fold gates apply (nothing moved).
  DESIGN proposals: none.
- Residuals (exact causes, each probed live): augment quad —
  foreign hunk in `test_augment_models.py` (retry post-land);
  video-worker quartet — needs an explicit fold target (mypy
  clear); audio remainder — no same-area target (all mypy
  clean); TUI app — stays solo by design; finalize/commit —
  owner `test_finalize_fastpath.py` foreign-dirty + hardening
  just-touched (retry once quiet); singletons — none left.
  Next passes use the same discipline (re-read live, check
  mtime/`git log`, keep assertion counts identical, move the
  gates.sh mypy entry with any fold that deletes a listed
  file).
