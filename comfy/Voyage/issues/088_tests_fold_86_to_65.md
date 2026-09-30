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
