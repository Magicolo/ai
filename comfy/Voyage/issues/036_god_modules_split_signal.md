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

## Progress log (2026-09-30, batch 12)

- Re-measured host `wc -l` (this pass, after both extractions):
  `supervisor.py` 2654, `media.py` 1631, `model_registry.py` 1264,
  `registry_records.py` 770, new `registry_film.py` 65,
  new `supervisor_commit_types.py` 63, `cli.py` 745 (seam, stable).
- Tree discipline per §9: `git diff --name-only` on owned targets was
  empty before every edit; concurrent tracks hold uncommitted foreign
  hunks in `TASK.md`, `BENCHMARKING.md`, `issues/031/035/070/079/093/
  127/152`, `cli_observe.py`, `scoreboard.py`, plus 2 untracked test
  files — all left intact, none inside either extraction region.
- Max TWO extractions reached via the 082 + 081 preference order; the
  036 remainder (cli_observe, media surface, video workers, test files)
  proves movable but stays recorded, not taken.
- 023 check: `backends._STREAMING_BACKENDS` is derived from
  `BACKEND_REGISTRY` but `supervisor.STREAMING_VIDEO_BACKENDS` is still
  a hand literal (`supervisor.py:120`, "mirrors" comment intact) — the
  unification has NOT landed, so routing maps were deliberately
  untouched per the batch brief.

## Resolution (2026-09-30, batch 12)

- Verdict: TRACKED — no 036-owned extraction this pass (quota filled by
  the 082 film family + 081 commit-types splits); signal table above
  stays current.
- Files changed: none under this tracker.
  Gate evidence: n/a (see 081/082 batch-12 entries for the two landed
  extractions: ruff + format + mypy strict clean on all 6 split files;
  18 split/neighbor tests green).
  DESIGN proposals (quoted, for the DESIGN owner — not applied here,
  file is out of scope): "No DESIGN text change proposed: this pass
  makes no 036-seam change; the split index, when created, should list
  `registry_film.py` (082 family pattern) and
  `supervisor_commit_types.py` (081 type pattern) alongside
  `registry_records.py` / `supervisor_proposal.py` /
  `supervisor_prefetch.py`."
   Residuals: `cli_observe.py` 734, `media.py` workers/media surface
   (open since batch 4), registry remaining 9 families, video-worker
   splits, then test-file clusters per 088 — each a future single-group
   pass with the same move-verbatim + facade + agreement-test discipline.

## Progress log (2026-09-30, batch 13 — two registry-family extractions)

- Re-measured host `wc -l` at pass start: `registry_records.py` 770
  (after batch 12's film split). `git diff --name-only` showed only
  out-of-scope files modified (`LTX2.md`, later a full concurrent
  flight: `config.py`, `cli_run_ops.py`, `cli_observe.py`, `augment.py`,
  several issues + tests) — `registry_records.py` itself was quiet, so
  both extractions went ahead in owned scope. `supervisor.py` was HOT
  (079 delete surface) and deliberately untouched; `cli_observe.py`,
  `scoreboard.py`, `model_registry.py`, and all longlive2 lines likewise
  untouched per the batch brief.
- Landed (1) realesrgan family (082 residual, next-smallest row):
  new `voyage/registry_realesrgan.py` (75L, DESIGN §§84-85) owns all 8
  `REALESRGAN_*` pins + `EXPECTED_REALESRGAN_SHA256` +
  `_record_realesrgan` + `_describe_realesrgan` verbatim;
  `registry_records.py` re-exports all 11 names via explicit-`as`
  self-aliases and carries move comments at the four old sites;
  `model_registry.py` untouched (its
  `from voyage.registry_records import ... REALESRGAN_*` chain holds
  through the facade). TDD: `tests/test_registry_realesrgan_split.py`
  written first (3 agreement tests mirroring the film suite) — watched
  fail on collection (`ModuleNotFoundError:
  voyage.registry_realesrgan`), then green after the move.
- Landed (2) inspector family (next-smallest decoupled row after
  realesrgan): new `voyage/registry_inspector.py` (75L, DESIGN
  §§43-44, 100, 132) owns all 7 `QWEN35_*` pins + `_record_inspector`
  + `_describe_inspector` verbatim; same facade + move-comment recipe
  (three old sites). One deliberate deviation from the film pattern:
  this row carries NO expected ingest hash (the manifest record holds
  no sha — same open residual as the CausVid checkpoint), so the new
  module drops the unused `sha256_file` import (ruff F401 would fire)
  and both docstrings record the residual. `cli_observe.py:91` reads
  `QWEN35_HF_REVISION` via call-time `getattr(model_registry, ...)` —
  facade-safe, no touch needed (and the file went foreign-modified
  mid-pass, left intact per §9). TDD: same red-then-green.
- Tree incident (foreign, recorded not fixed): mid-pass the image
  rebuild broke (`voyage doctor` → `ImportError:
  warn_if_deprecated_backend` from `voyage.config`, deleted by the
  079/config track while `cli_run_ops.py` still imported it). All
  in-container evidence below runs against the last green
  `voyage:latest` (2026-09-30 17:35) with the live tree bind-mounted
  (`$PWD:/app`, same invocation shape as `scripts/test.sh` minus the
  rebuild). The foreign group repaired the import chain mid-pass (the
  download parser/dispatch tests went red→green without any edit
  here); remaining reds are all on their surfaces (see Resolution).
- Max TWO reached; the 036 remainder (cli_observe branches, media
  surface, video workers, test-file clusters) stays recorded, not
  taken.

## Resolution (2026-09-30, batch 13)

- Verdict: TRACKED — no 036-seam extraction this pass (quota filled by
  the two 082 family splits below); signal table in the batch-12 entry
  stays current except `registry_records.py` 770→747.
- Files changed (owned scope only): `voyage/registry_realesrgan.py`
  (new, 75L), `voyage/registry_inspector.py` (new, 75L),
  `voyage/registry_records.py` (2 facades + 7 move comments, net
  −23L), `tests/test_registry_realesrgan_split.py` (new, 3 tests),
  `tests/test_registry_inspector_split.py` (new, 3 tests).
  `model_registry.py`, `supervisor.py`, `cli_observe.py`,
  `scoreboard.py` untouched.
  Gate evidence: in-container `ruff check` + `ruff format --check` +
  `mypy strict` clean on all 5 touched files; 15/15 split/agreement
  tests green (`test_registry_{inspector,realesrgan,film,records}_split`);
  neighbors 65 green (`test_augment_weight_loading`,
  `test_checkpoint_safety`, `test_director_models_dir`,
  `test_augment_models` download/verify paths). 3 reds, all foreign
  (079 longlive2-removal surface, verified by failure signature +
  foreign diff, none import from the touched files):
  `test_cuda_backends_include_augmentation_by_default[longlive2]` +
  `test_longlive.py::test_video_worker_module_map`
  (`ConfigurationError: unknown video backend 'longlive2' — removed
  (issue 079)`) + `test_models_verify_reports_augment_stacks`
  (`AttributeError: voyage.cli has no attribute
  'verify_longlive2_bf16'`). Full `gates.sh` not runnable (foreign
  image-build breakage above); per-file gates + scoped suites are the
  verdict.
  DESIGN proposals (quoted, for the DESIGN owner — not applied here,
  file is out of scope): "No DESIGN text change proposed: both new
  modules follow existing contracts (DESIGN §§84-85 augment weights;
  DESIGN §§43-44, 100, 132 visual inspector) and the issue-082
  move-verbatim + re-export + agreement-test convention; a future
  split index should list `registry_realesrgan.py` and
  `registry_inspector.py` alongside `registry_film.py`."
  Residuals: `registry_records.py` 747 with 7 families remaining
  (ltxv, causvid+wan21, sfx triple, director triple, audio pair,
  longlive+wan — longlive2 lines frozen for the 079 owner), plus the
  036 remainder above — each a future single-group pass.

## Addendum (2026-09-30, same pass — 079 longlive2 removal landed mid-pass)

- After the batch-13 edits above, the 079 group landed the longlive2
  removal inside owned-adjacent files: `model_registry.py` 1264→1188
  (longlive2 spec row + `LONGLIVE_*` imports gone) and
  `registry_records.py` 747→665 (longlive+wan pins + builders gone).
  Per §9 nothing of theirs was touched or reverted; my regions were
  re-verified intact after their landing (20 facade imports + all 7
  move comments present by grep).
- Failure set changed character with their landing (recorded, not
  fixed — their scope): `tests/test_registry_split.py` (batch-7
  agreement suite, not this pass's file) now fails 4 tests on stale
  longlive2 expectations (`EXPECTED_SPEC_KEYS` still lists
  `longlive2-bf16`; `model_registry` no longer exports
  `LONGLIVE_HF_REPO`) and `test_checkpoint_safety.py` fails 2
  longlive-named tests — all on the 079 removal surface. My 9
  agreement tests (`test_registry_{inspector,realesrgan,film}_split`)
  stay GREEN against the post-removal tree, and per-file gates stay
  clean on all 4 fully-owned files plus joint `registry_records.py`
  (`ruff check` + `ruff format --check` + `mypy strict`). Updating
  the stale `test_registry_split.py` expectations belongs to the 079
  owner, not this track.
- Never committed; `000_INDEX.md` / `DESIGN.md` / `AGENTS.md`
  untouched (verified via `git diff --name-only`).

## Progress log (2026-10-01, two registry-family extractions)

- Re-measured host `wc -l` at pass start (in-container
  `voyage:latest`, CPU-only): `registry_records.py` 665 (after the
  079 longlive2 removal); after both extractions 646. Full signal
  table this pass: `supervisor.py` 2624, `media.py` 1631,
  `video_causvid.py` 1226, `model_registry.py` 1188,
  `video_ltxv.py` 1159, `tui.py` 1146, `config.py` 1028,
  `cli.py` 736 (seam, stable point), new `registry_audio.py` 105,
  new `registry_ltxv.py` 84, `registry_film.py` 65,
  `registry_realesrgan.py` 75, `registry_inspector.py` 75,
  `test_tui_app.py` 989, `test_causvid_worker.py` 774.
- Tree discipline per §9: `git diff --name-only` on owned targets
  was empty before every edit; `Voyage/` quiet except
  `M Voyage/LTX2.md` (left intact, never touched); concurrent
  `../tango/Tango` untracked, never touched. `supervisor.py`,
  `media.py`, video workers, `model_registry.py`, `cli_observe.py`
  deliberately untouched per the brief (foreign/hot or owned
  elsewhere).
- Landed (1) LTXV family (082 residual, smallest decoupled row):
  new `voyage/registry_ltxv.py` (84L, DESIGN Phase 7, §§84-85)
  owns all 13 `LTXV_*` pins + 2 `EXPECTED_LTXV_*` hashes +
  `_record_ltxv` + `_describe_ltxv` verbatim;
  `registry_records.py` re-exports all 17 names via explicit-`as`
  self-aliases and carries move comments at the four old sites;
  `model_registry.py` untouched (its
  `from voyage.registry_records import ... LTXV_*` chain holds
  through the facade). TDD: `tests/test_registry_ltxv_split.py`
  written first (3 agreement tests mirroring the film suite) —
  watched fail on collection (`ModuleNotFoundError:
  voyage.registry_ltxv`), then green after the move.
- Landed (2) audio family (next-smallest decoupled row): new
  `voyage/registry_audio.py` (105L, DESIGN §§6, 37) owns all 13
  `ACE_*` pins + `_ACE_CHECKPOINTS_RELATIVE` +
  `_ACE_LM_RELATIVE` + `_record_audio` + `_describe_audio`
  verbatim; same facade + move-comment recipe (four old sites).
  One deliberate deviation from the film pattern: this row
  carries NO expected ingest hash (manifest record holds no sha
  — same open residual as inspector/CausVid), so the new module
  drops the unused `sha256_file` import (ruff F401 would fire)
  and both docstrings record the residual. TDD: same
  red-then-green (`ModuleNotFoundError: voyage.registry_audio`).
- Max TWO reached; the 036 remainder stays recorded, not taken:
  supervisor commit methods, `media.py` workers/media surface
  (open since batch 4), video-worker splits (`video_ltxv.py`
  1159 / `video_causvid.py` 1226), then the two big test files by
  area (`test_tui_app.py` 989 / `test_causvid_worker.py` 774).

## Resolution (2026-10-01, two registry-family extractions)

- Verdict: TRACKED — no 036-seam extraction this pass (quota filled by
  the two 082 family splits above); signal table in the entry above
  stays current except `registry_records.py` 665→646 plus new
  `registry_ltxv.py` 84 + `registry_audio.py` 105 in the split index.
- Files changed (owned scope only): `voyage/registry_ltxv.py`
  (new, 84L), `voyage/registry_audio.py` (new, 105L),
  `voyage/registry_records.py` (2 facades + 8 move comments, net
  −19L), `tests/test_registry_ltxv_split.py` (new, 3 tests),
  `tests/test_registry_audio_split.py` (new, 3 tests).
  `model_registry.py`, `supervisor.py`, `media.py`,
  video workers, `cli_observe.py` untouched.
  Gate evidence: in-container `ruff check` + `ruff format --check` +
  `mypy strict` clean on all 5 touched files; 15/15 split/agreement
  tests green (`test_registry_{ltxv,audio,film,realesrgan,inspector}_split`);
  neighbors 69 green (`test_registry_split`,
  `test_registry_pins`, `test_augment_models`,
  `test_augment_weight_loading`, `test_checkpoint_safety`,
  `test_director_models_dir`) + 60 on the worker-adjacent set
  (`test_ltxv` + `test_causvid_prep` + `test_sfx_contract`).
  Full `gates.sh` not run (quota is per-file gates + scoped suites).
  DESIGN proposals (quoted, for the DESIGN owner — not applied here,
  file is out of scope): "No DESIGN text change proposed: both new
  modules follow existing contracts (DESIGN Phase 7 LTXV weights;
  DESIGN §§6, 37 ACE-Step music stack) and the issue-082
  move-verbatim + re-export + agreement-test convention; a future
  split index should list `registry_ltxv.py` and `registry_audio.py`
  alongside `registry_film.py` / `registry_realesrgan.py` /
  `registry_inspector.py`."
  Residuals: `registry_records.py` 646 with 3 families remaining
  (director triple incl. shared MINILM, causvid+wan21, sfx triple),
  plus the 036 remainder above (supervisor commit methods,
  `media.py` surface, video workers, test clusters) — each a future
  single-group pass. Manifest-race fix + supervisor-side derivation
  stay with their owners (recorded in 082, not touched here).
