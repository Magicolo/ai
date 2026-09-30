# 036 — God modules: `cli.py` 2105 / `supervisor.py` 1902 lines vs §12 ~500-line split signal

- Severity: MEDIUM
- Files: `voyage/cli.py:1` (2105), `voyage/supervisor.py:1` (1902), `voyage/model_registry.py:1` (1368), `voyage/workers/video_longlive.py:1` (1264), `voyage/workers/video_causvid.py:1` (1157), `voyage/media.py:1` (1116), `voyage/tui.py:1` (1078); tests `tests/test_tui_app.py:1` (943), `tests/test_causvid_worker.py:1` (748)
- Area: structure
- Overlaps with: 080/081 (cli/supervisor split execution — this file is the signal, those are the splits)

## Description

AGENTS.md §12 sets a ~500-line module-split soft signal with god-module
watch on `cli/supervisor`/video workers. Seven shipped modules exceed 2-4x
that; `media.py` also carries the open "workers/media surface stays open"
split from batch 4. Large-file reviewability, `PLR0912/0915` and `C901` all
concentrate here.

## Rationale

Size correlates with branch/statement-count violations and merge contention
(AGENTS.md §9 warns concurrent agents collide in `supervisor.py`). The
standardization pass already split once (batch 4: `video_common`, `hashing`,
`BackendName` registry) — the remainder is tracked nowhere.

## Live evidence (re-verified 2026-09-30)

Host `wc -l` today (re-verified 2026-09-30, drifted up from the Track-C sweep by concurrent edits):

```
2105 voyage/cli.py
1902 voyage/supervisor.py
1368 voyage/model_registry.py
1264 voyage/workers/video_longlive.py
1157 voyage/workers/video_causvid.py
1116 voyage/media.py
1078 voyage/tui.py
975 voyage/workers/video_ltxv.py
940 voyage/config.py
```

(Track-C sweep recorded 2030 for `cli.py`; current tree reads 2105 — drift
is growth, same finding.) Sweep `ALL` stats (preserved): `PLR0912 13
too-many-branches, PLR0915 11 too-many-statements, C901 16
complex-structure`.

## Repro

```bash
wc -l voyage/*.py voyage/workers/*.py | sort -rn | head
```

## Fix candidates

Split `cli.py` by verb group (generate/run/finalize vs models/doctor),
extract `supervisor` commit/augment helpers, extract `model_registry`
per-backend record builders; split `test_tui_app.py` /
`test_causvid_worker.py` by area. No behavior change, DESIGN §-ref'd new
modules.

## Fold note (2026-09-30, append-only — this tracker stays open)

080/082/085 landed as the first three executions of this signal; this file
is folded as their tracker (not deleted) until the remaining splits land.

- Landed: `cli.py` 2403 → **735L** (080, verbs in 10 modules ≤365L;
  parsers retained as the seam by design); `model_registry.py` 1543 →
  **1270L** (082, pins/builders in `registry_records.py` 767L;
  per-family tables + manifest race remain); 085 collapses (epsilon,
  fallbacks, reserved leaf, TOML fallbacks) + agreement tests for the
  rest. Evidence: `tests/test_cli_split.py` (13) +
  `tests/test_registry_split.py` (7) + `tests/test_single_source.py` (8);
  15/15 `--help` goldens identical; live `init/status/validate` smoke exit 0.
- Still above the ~500L signal (host `wc -l` today): `supervisor.py`
  2538, `model_registry.py` 1270, `video_longlive.py` 1301,
  `video_causvid.py` 1157, `media.py` 1404, `tui.py` 1080,
  `video_ltxv.py` 1115, `config.py` 1023, `cli.py` 735 (seam, stable
  point), `test_tui_app.py` 943, `test_causvid_worker.py` 748.
- Next in rank order: supervisor commit/augment-helper extraction (needs a
  quiet tree — concurrently modified today), `media.py` workers/media
  surface (open since batch 4), per-family registry tables (082
  residual), video-worker splits, then the two test files by area.
  Procedure per split: re-verify premises live, fail-first surface tests,
  verbatim moves + DAG check + goldens, seam-dispatch rule for patched
  names (see 080 Resolution), gates on touched files only.

## Refs

- AGENTS.md §12 soft signals ("~500-line module split signal, god-module
  watch, dead-code removal on sight").
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §6.

## Progress log (2026-09-30, this pass — one safe extraction)

- Re-measured host `wc -l` (batch-9 tree; `supervisor.py` 2760→2701 via
  foreign edits, never touched here): `supervisor.py` 2701, `media.py`
  1631, `video_longlive.py` 1346, `model_registry.py` 1264,
  `video_causvid.py` 1226, `video_ltxv.py` 1159, `tui.py` 1155,
  `config.py` 1037, `cli_observe.py` 769, `registry_records.py` 767,
  `cli.py` 745.
- Tree was quiet at pass start (`git diff --name-only` clean; concurrent
  batch-9 + uncommitted foreign edits landed mid-pass in other files —
  left intact per §9), so the one extraction went ahead in owned scope:
  the `cmd_inspect` scoreboard branch out of `cli_observe.py` (769, above
  the signal) into new `voyage/cli_scoreboard.py` (`render_scoreboard`,
  DESIGN §59, 54 lines) — verbatim move, `cmd_inspect` keeps its name
  and delegates, so the 080 seam (`cli.cmd_inspect is
  cli_observe.cmd_inspect`) holds.
- TDD: `tests/test_cli_scoreboard.py` written first (4 tests: empty run,
  no-visual row, `0.500(+0.000)` metric cells, trailing `final.mp4`
  line) — watched fail on collection (`ModuleNotFoundError`), then green
  after the move. One real catch on the way: the first cut imported
  `scoreboard_rows` at module top, which froze the reference and broke
  `test_inspect_scoreboard_tolerates_unformattable_cells` (it patches
  `voyage.scoreboard.scoreboard_rows`) — fixed by restoring the original
  call-time import inside `render_scoreboard` (080 seam-dispatch rule),
  all green after.
- Next splits in rank order (all owned elsewhere — recorded, not taken):
  supervisor commit/augment-helper extraction (foreign-owned, needs a
  quiet tree), `media.py` workers/media surface (open since batch 4),
  registry per-family tables (082 residual — this pass migrated their
  types, not their location), video-worker splits, then the two big
  test files by area.

## Resolution (2026-09-30, this pass)

- Verdict: PARTIALLY RESOLVED — one extraction landed
  (`cli_observe.py` 769→741 + new `cli_scoreboard.py` 54).
- Files changed: `voyage/cli_scoreboard.py` (new),
  `voyage/cli_observe.py` (branch → delegation),
  `tests/test_cli_scoreboard.py` (new, 4 tests). Gate evidence:
  in-container `mypy` strict + `ruff check` + `ruff format --check`
  clean on all 3; 39 tests green (`test_cli_scoreboard`,
  `test_scoreboard`, `test_cli_split` incl. 080 seam identity,
  `test_issue_142_inspect_failsoft` incl. the seam-patch test,
  `test_inspect_metrics_fps_029`, `test_cli_tui_split`) + 42 CLI
  neighbors (`test_cli_group_a`, `test_cli_benchmark_sfx_augment`,
  `test_observability_rank2`) + full `mypy voyage` (65 files) +
  `ruff check .` whole-tree green.
  DESIGN proposals (quoted, for the DESIGN owner — not applied here,
  file is out of scope): "No DESIGN text change proposed: the new
  module follows the existing DESIGN §59 scoreboard contract and the
  issue-080 verb-module seam convention; a future split index could
  list `cli_scoreboard.py` alongside the other `cli_*` fragments."
   Residuals: tracker table above (supervisor/media/workers/test
   splits still open under their owners).

## Progress log (2026-09-30, two-extraction pass — cli_inspect_metrics + supervisor_prefetch)

- Re-measured host `wc -l` (this pass, after both extractions):
  `supervisor.py` 2686, `media.py` 1631, `video_longlive.py` 1346,
  `model_registry.py` 1264, `video_causvid.py` 1226,
  `video_ltxv.py` 1159, `tui.py` 1146, `config.py` 1037,
  `cli_observe.py` 734, `cli.py` 745 (seam, stable point),
  new `cli_inspect_metrics.py` 31, `cli_scoreboard.py` 54,
  new `supervisor_prefetch.py` 43, `supervisor_proposal.py` 91.
- Tree discipline per §9: `git diff --name-only` on the four target
  files was empty before every edit; concurrent tracks hold
  uncommitted foreign hunks in `audio/beat.py`, `audio/planner.py`,
  `model_registry.py`, `tests/test_rhythm.py`,
  `tests/test_beat_quantize_ties_121.py`,
  `tests/test_augment_contract_166.py`, `issues/024_*` — all left
  intact, none inside either extraction region (verified: supervisor
  diff shows only the facade + deletion).
- Landed (1) `cli_observe.py` 741→734 + new `cli_inspect_metrics.py`
  31L: the `cmd_inspect -- metrics` branch moved verbatim into
  `render_inspect_metrics` (DESIGN §59; top-level imports from
  `cli_status`/`logrotate` preserved, no patched leaves so no
  call-time freeze needed); `cmd_inspect` keeps its name and
  delegates (080 seam `cli.cmd_inspect is cli_observe.cmd_inspect`
  holds); the now-unused `iter_metric_files` top import was deleted
  (zero importers from `cli_observe`, verified via rg).
  TDD: `tests/test_cli_inspect_metrics.py` written first (4 tests:
  empty run, rotated-files count + span, last-five trim, seam
  delegation) — watched fail on collection (`ModuleNotFoundError`),
  then green after the move. One format catch on the way (ruff
  wanted two `def` lines joined) — fixed, all green after.
- Landed (2) `supervisor.py` 2707→2686 + new
  `supervisor_prefetch.py` 43L: `summarize_prefetch_outcome`
  moved verbatim (docstring incl. 136/168 third-outcome note);
  `supervisor.py` carries the explicit-`as` facade re-export
  (mypy single-source pattern from the proposal pass) + a move
  comment at the old site. Routing (`VIDEO/AUDIO_WORKER_MODULES`,
  `STREAMING_VIDEO_BACKENDS` — issue 023's area) deliberately STAYS.
  TDD: `tests/test_supervisor_prefetch_helpers.py` written first (5
  tests: facade single-source, empty, mixed, invalidated-excluded,
  all-miss) — `ModuleNotFoundError` red, then green. The
  invalidated-exclusion pin is new coverage (old suite never
  asserted it).
- Skipped (3) test-file cluster fold: max TWO reached; remaining
  088 clusters (adapter triple, augment quad, audio validators, TUI
  trio, video-worker quartet, finalize/commit merges, leftover
  singletons) stay open — one per pass with the same discipline.

## Resolution (2026-09-30, two-extraction pass)

- Verdict: PARTIALLY RESOLVED — two extractions landed
  (`cli_observe.py` 741→734 + new `cli_inspect_metrics.py` 31;
  `supervisor.py` 2707→2686 + new `supervisor_prefetch.py` 43).
- Files changed: `voyage/cli_inspect_metrics.py` (new),
  `voyage/cli_observe.py` (branch → delegation, unused import
  dropped), `tests/test_cli_inspect_metrics.py` (new, 4 tests),
  `voyage/supervisor_prefetch.py` (new),
  `voyage/supervisor.py` (facade re-export + block → move
  comment, net −21L),
  `tests/test_supervisor_prefetch_helpers.py` (new, 5 tests).
  Gate evidence: in-container `mypy` strict + `ruff check` +
  `ruff format --check` clean on all 6; 58 tests green for (1)
  (`test_cli_inspect_metrics`, `test_cli_scoreboard`,
  `test_cli_split`, `test_inspect_metrics_fps_029`,
  `test_issue_142_inspect_failsoft`, `test_scoreboard` = 33 +
  `test_cli_benchmark_sfx_augment` + `test_observability_rank2` =
  25) and 77 for (2) (`test_supervisor_prefetch_helpers`,
  `test_prefetch_summary`, `test_prefetch_shutdown`,
  `test_three_captions`, `test_sfx_contract` = 31 +
  `test_failure_policy` + `test_commit_hardening` +
  `test_generation_stack` + `test_supervisor_lifecycle` = 46),
  all CPU-only.
  DESIGN proposals (quoted, for the DESIGN owner — not applied here,
  file is out of scope): "No DESIGN text change proposed: both new
  modules follow existing contracts (DESIGN §59 inspect views;
  DESIGN §§73/68 supervisor lifecycle + soak) and the issue-080
  verb-module / issue-081 move-verbatim + re-export + agreement-test
  conventions; a future split index could list
  `cli_inspect_metrics.py` alongside the other `cli_*` fragments
  and `supervisor_prefetch.py` alongside `supervisor_proposal.py`."
  Residuals: supervisor commit/augment-helper remainder, `media.py`
  workers/media surface (open since batch 4), registry per-family
  tables (082 residual), video-worker splits, then test-file
  clusters per 088 — each a future single-group pass.
