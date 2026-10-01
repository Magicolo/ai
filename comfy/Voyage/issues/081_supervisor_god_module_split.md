# 081 — Split `supervisor.py` god module (1894L, commit + lifecycle + audio-cover)

- Severity: HIGH (structure)
- File: `voyage/supervisor.py:1` (1902 lines, 45 fns — sweep said 1894)
- Area: structure — supervisor decomposition

## Description

`supervisor.py` carries lifecycle state machine + transactional segment commit + audio coverage + video render + director accept in one file. Load-bearing giants: `_ensure_audio_coverage` 150L, `_commit_segment` 137L, `_accept_director_decision` 135L, `_render_video` 117L, `_propose_segment` + `run_segments` ownership. Side seams: `sha256_file as sha256_file` re-export shim (`:50`, import `hashing` directly), `STREAMING_VIDEO_BACKENDS` hand literal (`:96`) vs registry-derived `backends._STREAMING_BACKENDS` (`backends.py:82-88`, ownership open, issue 023), `VIDEO/AUDIO_WORKER_MODULES` maps (`:88-108`) with 3 near-identical unknown-backend errors, flat deterministic-backend payload compat block (`:859`), `LEGACY_MIGRATION_REMOVE_AFTER` threading (`:1460`), `run_id` legacy (`:469`).

## Rationale

Same §12 split-signal case as cli (4x over). Highest concurrent-edit collision file (§9 warning). Commit-pipeline reviewers must load lifecycle + audio + render context to review one path.

## Live evidence

- `wc -l voyage/supervisor.py` → 1902; `grep -n "^def \|^class " voyage/supervisor.py` → 45 fns
- `:88-108` worker-module maps + streaming tuple; `:242,595,605,666` bare `except Exception` samples (sweep cited `:373,570,599,1331` — drifted)
- `:859` flat deterministic payload comment; `:1460` legacy_path threading
- `backends.py:82-88` docstring: "supervisor track owns unifying this with supervisor.STREAMING_VIDEO_BACKENDS"

## Repro

```bash
wc -l voyage/supervisor.py; grep -n "def _ensure_audio_coverage\|def _commit_segment\|def _accept\|def _render_video\|STREAMING_VIDEO_BACKENDS\|VIDEO_WORKER_MODULES" voyage/supervisor.py
```

## Fix candidates

1. Extract commit pipeline (`commit.py`: propose/render/cover/commit/accept) vs lifecycle (`run_segments`/state/locks) vs audio-coverage; keep `supervisor.py` as facade re-exporting for one release, then cut over.
2. Delete `sha256_file` re-export; derive streaming set from `BACKEND_REGISTRY` (see 083); merge 3 worker-module maps into one table or registry derivation.
3. Gate: `gates.sh` green + crash-matrix/commit/integration suites (`test_crash_matrix`, `test_commit_hardening`, `test_commit_split`, `test_integration`) green.

## Refs

- Issues 023 (streaming unification owner), 025, 036; `voyage/backends.py:74-89`, `voyage/config.py:88-184`

## Progress log (2026-09-30, single-group extraction pass)

- Pre-checks live in-container
  (`docker run --rm -v $PWD/Voyage:/app -w /app voyage:latest`):
  `wc -l voyage/supervisor.py` → 2760 (was 1902 at issue filing —
  growth is commit-pipeline features, not drift); `git diff --name-only`
  showed `supervisor.py` CLEAN (concurrent tracks touch `cli_*`,
  `model_registry` typing, `scoreboard`, `augment_worker` — none inside
  the extraction region), so a quiet-region extraction is safe. Did NOT
  attempt the full decomposition (per task: one coherent group only).
- TDD failing-first: wrote
  `tests/test_supervisor_proposal_helpers.py` (5 agreement/behavior
  tests) BEFORE the new module — in-container collection failed with
  `ModuleNotFoundError: No module named 'voyage.supervisor_proposal'`
  (red), then created the module + re-export until green.
- Extraction: new `voyage/supervisor_proposal.py` (91L, DESIGN §§73,
  18.2) owns the contiguous director-proposal pure-helper block
  (`previous_transition_captions`, `effective_music_caption`,
  `effective_video_stages`, `_token_counts` — ex-`supervisor.py:264-329`,
  moved verbatim except one `# noqa: BLE001` on the intentional
  best-effort `except Exception`); `supervisor.py` (2760→2701L) imports
  and re-exports all four names at the top, drops the now-unused
  `load_transition` import, and carries a move comment at the old site.
  Routing (`VIDEO/AUDIO_WORKER_MODULES`, `STREAMING_VIDEO_BACKENDS`,
  `audio/video_worker_module` — issue 023's unification area) and
  prefetch summary (`summarize_prefetch_outcome`) deliberately STAY for
  their owning passes. `load_transition`/`EvolutionDecision` remain used
  elsewhere in `supervisor.py`, so no other import churn.
- Gate evidence (in-container `voyage:latest`): new agreement suite 5
  passed; existing importers `test_prefetch_summary + test_sfx_contract
  + test_three_captions` 29 passed; related suites `test_failure_policy
  + test_crash_matrix + test_commit_hardening + test_generation_stack`
  49 passed; `test_commit_split + test_hashing + test_tui_state +
  test_tui` 91 passed. Per-file gates: `ruff check` clean (one BLE001
  handled by inline noqa — the module cannot inherit
  `supervisor.py`'s per-file `BLE` ignore without touching shared
  `pyproject.toml`, owned by the toolchain track), `ruff format
  --check` clean on all 3 files, `mypy strict` clean on both modules.
  Full `gates.sh` follow-up: mypy flagged the facade re-export
  (`supervisor` does not explicitly export the moved names) in
  `test_three_captions`/`test_sfx_contract` — fixed by spelling the
  facade import with explicit `as` self-aliases (ruff isort then split
  it into four single-name `from` statements; canonical, accepted).
  After that fix: ruff + format + mypy clean on all own files and all
  suites above re-green. Full-tree `gates.sh` still reports 3 mypy
  errors in `tests/test_checkpoint_safety.py:105` — FOREIGN
  (concurrent track's `JsonValue` return-type migration in
  `model_registry.py`/`registry_records.py`; that test file is
  untouched by this pass, verified via empty `git diff` on it).

## Resolution (2026-09-30, single-group extraction pass)

- Verdict: single-group split landed; full god-module decomposition
  remains open. Files changed: `voyage/supervisor_proposal.py` (new,
  91L), `voyage/supervisor.py` (top re-export import + delete moved
  block + move comment, net −59L), `tests/test_supervisor_proposal_helpers.py`
  (new, 5 agreement/behavior tests).
- DESIGN proposals: "Record the split in DESIGN §73 (supervisor owns
  lifecycle and commit state): director-proposal pure helpers
  (`previous_transition_captions`, `effective_music_caption`,
  `effective_video_stages`, `_token_counts`) now live in
  `voyage/supervisor_proposal.py` with `voyage/supervisor.py` as the
  re-export facade; future extractions (routing/unification with
  `backends._STREAMING_BACKENDS` per issue 023, prefetch/gauge helpers,
  tape helpers, stage-timing) follow the same move-verbatim +
  re-export + agreement-test pattern, one group per pass."
- Residuals: full decomposition (commit pipeline vs lifecycle vs
  audio-coverage per fix candidate 1; `sha256_file` re-export shim;
  streaming-set derivation per 023/083; worker-module map merge;
  deterministic-payload compat block; legacy-migration threading;
  `run_id` legacy) — each a future single-group pass with its own
  agreement tests. No concurrent collision met (supervisor region was
  quiet); `model_registry.py` typing hunks from another track coexist
  untouched.

## Progress log (2026-09-30, prefetch-extraction pass)

- Pre-checks live: `wc -l voyage/supervisor.py` → 2707 at pass
  start (was 2701 at the proposal pass — +6 is the facade import
  block growth, not drift); `git diff --name-only --
  Voyage/voyage/supervisor.py` empty before BOTH edits (facade
  import, then block deletion), so the quiet-region rule held.
  Concurrent tracks hold uncommitted foreign hunks in
  `audio/beat.py`, `audio/planner.py`, `model_registry.py`,
  `tests/test_rhythm.py`, `tests/test_beat_quantize_ties_121.py`,
  `tests/test_augment_contract_166.py` — none inside the
  extraction region (final `git diff` on `supervisor.py` shows
  only the facade + deletion, 8 insertions / 36 deletions across
  both files with `cli_observe.py`).
- TDD failing-first: wrote
  `tests/test_supervisor_prefetch_helpers.py` (5 agreement/behavior
  tests: facade single-source, empty, mixed, invalidated-excluded,
  all-miss) BEFORE the new module — in-container collection failed
  with `ModuleNotFoundError: No module named
  'voyage.supervisor_prefetch'` (red), then created the module +
  re-export until green.
- Extraction: new `voyage/supervisor_prefetch.py` (43L, DESIGN
  §§73, 68) owns `summarize_prefetch_outcome` verbatim
  (ex-`supervisor.py:232-257`, docstring incl. the 136/168
  third-outcome note); `supervisor.py` (2707→2686L) imports and
  re-exports via the explicit-`as` self-alias (mypy single-source
  pattern from the proposal pass — no follow-up fix needed this
  time) and carries a move comment at the old site. Routing
  (`VIDEO/AUDIO_WORKER_MODULES`, `STREAMING_VIDEO_BACKENDS` —
  issue 023's unification area) deliberately STAYS for its owning
  pass. `cli_observe.py:645` resolves the name through the facade
  at call time (function-local import), so the soak report path is
  unchanged.
- Gate evidence (in-container `voyage:latest`, CPU-only): new
  agreement suite 5 passed; existing importers
  `test_prefetch_summary + test_prefetch_shutdown +
  test_three_captions + test_sfx_contract` 26 passed (31 with the
  new suite); commit-path neighbors `test_failure_policy +
  test_commit_hardening + test_generation_stack +
  test_supervisor_lifecycle` 46 passed. Per-file gates: `ruff
  check` + `ruff format --check` + `mypy strict` clean on all 3
  files. No full-tree `gates.sh` run (foreign hunks in
  `model_registry.py`/audio would color it); touched-file gates +
  the 77 related tests are the verdict.

## Resolution (2026-09-30, prefetch-extraction pass)

- Verdict: second single-group split landed; full god-module
  decomposition remains open. Files changed:
  `voyage/supervisor_prefetch.py` (new, 43L),
  `voyage/supervisor.py` (facade re-export + block → move
  comment, net −21L),
  `tests/test_supervisor_prefetch_helpers.py` (new, 5
  agreement/behavior tests incl. the first invalidated-exclusion
  pin).
- DESIGN proposals: "No DESIGN text change proposed: the new
  module follows the existing DESIGN §§73/68 contract (supervisor
  lifecycle + soak report) and the issue-081 move-verbatim +
  re-export + agreement-test pattern; future extractions (routing
  unification with `backends._STREAMING_BACKENDS` per issue 023,
  gauge helpers, tape helpers, stage-timing) follow the same
  pattern, one group per pass."
- Residuals: full decomposition minus prefetch (commit pipeline vs
  lifecycle vs audio-coverage per fix candidate 1;
  `sha256_file` re-export shim; streaming-set derivation per
  023/083; worker-module map merge; deterministic-payload compat
  block; legacy-migration threading; `run_id` legacy) — each a
  future single-group pass with its own agreement tests. No
  concurrent collision met (supervisor region quiet at both
  edits); foreign `model_registry.py`/audio/rhythm hunks coexist
  untouched.

## Progress log (2026-09-30, batch 12)

- Pre-checks live: `wc -l voyage/supervisor.py` → 2688 at pass start;
  `git diff --name-only -- voyage/supervisor.py` empty before BOTH
  edits (facade import, then block deletion + unused-import cleanup),
  so the quiet-region rule held. 023 derivation verified NOT landed
  (`supervisor.STREAMING_VIDEO_BACKENDS` still hand literal) — routing
  maps deliberately untouched.
- TDD failing-first: wrote `tests/test_supervisor_commit_types.py`
  (2 agreement tests: facade identity + field-name stability) BEFORE
  the new module — in-container collection failed with
  `ModuleNotFoundError: No module named
  'voyage.supervisor_commit_types'` (red), then created the module +
  re-export until green.
- Extraction: new `voyage/supervisor_commit_types.py` (63L, DESIGN
  §73) owns `ProposedSegment` / `RenderedVideo` / `CoveredAudio`
  verbatim (ex-`supervisor.py:195-238`, docstrings intact);
  `supervisor.py` (2688→2654L) imports and re-exports via the
  explicit-`as` self-alias pattern and carries a move comment at the
  old site. Cleanup: removed now-unused `NamedTuple` +
  `PromptPlan` imports (ruff F401, verified still unused via rg —
  `EvolutionDecision`/`AudioPlan` stay, used at 8+ sites).
- Gate evidence (in-container `voyage:latest`, CPU-only): new suite
  2 passed; neighbors `test_commit_split` +
  `test_supervisor_proposal_helpers` +
  `test_supervisor_prefetch_helpers` + `test_generation_stack`
  30 passed. Per-file gates: `ruff check` + `ruff format --check` +
  `mypy strict` clean on all 3 files (`supervisor.py`,
  `supervisor_commit_types.py`, `test_supervisor_commit_types.py`).

## Resolution (2026-09-30, batch 12)

- Verdict: third single-group split landed; full god-module
  decomposition remains open. Files changed:
  `voyage/supervisor_commit_types.py` (new, 63L),
  `voyage/supervisor.py` (facade re-export + block → move comment +
  2 unused imports dropped, net −34L),
  `tests/test_supervisor_commit_types.py` (new, 2 agreement tests).
- DESIGN proposals: "No DESIGN text change proposed: the new module
  follows the existing DESIGN §73 contract (supervisor lifecycle and
  commit state) and the issue-081 move-verbatim + re-export +
  agreement-test pattern; a future split index should list
  `supervisor_commit_types.py` alongside `supervisor_proposal.py` /
  `supervisor_prefetch.py`."
- Residuals: full decomposition minus proposal/prefetch/commit-types
  (commit pipeline methods vs lifecycle vs audio-coverage per fix
  candidate 1; `sha256_file` re-export shim; streaming-set derivation
  per 023/083 — still open, verified this pass; worker-module map
  merge; deterministic-payload compat block; legacy-migration
  threading; `run_id` legacy) — each a future single-group pass.

## Progress log (2026-09-30, batch 13 — record only, no supervisor work)

- No supervisor-side extraction this pass, by brief: `supervisor.py`
  is HOT (another group owns the 079 delete surface including
  supervisor hunks — the tree shows heavy concurrent flight in
  `config.py`/`cli_*`/`augment.py` mid-pass). Recorded, not taken:
  the commit-pipeline/augment-helper remainder from the batch-12
  residual stands unchanged. This track's batch-13 quota went to two
  082 registry-family splits (realesrgan + inspector, see 082); 036
  carries the pass verdict and the foreign-tree incident log.

## Progress log (2026-10-01, tape-extraction pass)

- Pre-checks live: `wc -l voyage/supervisor.py` → 2650 at pass start;
  `git diff --name-only -- voyage/supervisor.py` empty before BOTH edits
  (facade import, then method delegation), so the quiet-region rule held.
  Concurrent tracks hold foreign hunks in `pyproject.toml` (079 ignore
  removal), `tests/test_finalize_fastpath.py`,
  `tests/test_media_robustness_rank2.py`, untracked
  `tests/test_registry_audio_split.py` /
  `tests/test_registry_ltxv_split.py`, plus many `issues/*` edits —
  none inside the extraction region (final `git diff` on
  `supervisor.py` shows only the facade + delegation, 5 insertions /
  31 deletions). Routing maps (`VIDEO/AUDIO_WORKER_MODULES`,
  `STREAMING_VIDEO_BACKENDS` — issue 023 area), commit-pipeline core,
  and audio-coverage deliberately untouched.
- TDD failing-first: wrote `tests/test_supervisor_tape_helpers.py` (6
  agreement/behavior tests: facade single-source, method delegation,
  non-JSON adopt, missing-keys adopt, absent-tail adopt,
  mismatch-reject) BEFORE the new module — in-container collection
  failed with `ModuleNotFoundError: No module named
  'voyage.supervisor_tape'` (red), then created the module + re-export
  until green.
- Extraction: new `voyage/supervisor_tape.py` (51L, DESIGN §§73, 27)
  owns `tape_tail_sha_matches` verbatim
  (ex-`supervisor.py:744-775` staticmethod body, docstring intact);
  `supervisor.py` (2650→2624L) imports and re-exports via the
  explicit-`as` self-alias pattern and keeps
  `Supervisor._tape_tail_sha_matches` as a one-line delegating
  `@staticmethod` wrapper (callers via `self.` unchanged). No
  unused-import cleanup (`json`/`JsonValue`/`sha256_file` all still
  used elsewhere — verified via rg).
- Gate evidence (in-container `voyage:latest`, CPU-only): new agreement
  suite 6 passed; neighbors `test_supervisor_proposal_helpers` +
  `test_supervisor_prefetch_helpers` + `test_supervisor_commit_types`
  + `test_tape_trust_123_171` + `test_recovery` +
  `test_supervisor_lifecycle` 36 passed; commit-path neighbors
  `test_commit_hardening` + `test_failure_policy` +
  `test_generation_stack` + `test_prefetch_summary` 47 passed (pre-fold).
  Per-file gates: `ruff check` + `ruff format --check` + `mypy strict`
  clean on all 3 files (`supervisor.py`, `supervisor_tape.py`,
  `test_supervisor_tape_helpers.py`; one `ruff format` reflow on the
  new test file). DESIGN-ref check passes (`DESIGN §§73, 27` in the
  first 15 lines). No full-tree `gates.sh` run (foreign hunks in
  `pyproject.toml`/tests would color it); touched-file gates + the 83
  related tests are the verdict.

## Resolution (2026-10-01, tape-extraction pass)

- Verdict: fourth single-group split landed; full god-module
  decomposition remains open. Files changed:
  `voyage/supervisor_tape.py` (new, 51L),
  `voyage/supervisor.py` (facade re-export + staticmethod → delegation,
  net −26L),
  `tests/test_supervisor_tape_helpers.py` (new, 6 agreement/behavior tests).
- DESIGN proposals: "No DESIGN text change proposed: the new module
  follows the existing DESIGN §§73/27 contract (supervisor lifecycle
  and recovery-tape discovery) and the issue-081 move-verbatim +
  re-export + agreement-test pattern; a future split index should list
  `supervisor_tape.py` alongside `supervisor_proposal.py` /
  `supervisor_prefetch.py` / `supervisor_commit_types.py`."
- Residuals: full decomposition minus proposal/prefetch/commit-types/tape
  (commit pipeline methods vs lifecycle vs audio-coverage per fix
  candidate 1; `sha256_file` re-export shim; streaming-set derivation
  per 023/083 — still open; worker-module map merge;
  deterministic-payload compat block; legacy-migration threading;
  `run_id` legacy; remaining tape helpers `_latest_recovery_tape` /
  `_resume_video_worker` are stateful and stay for a future pass) —
  each a future single-group pass.

## Progress log (2026-10-01, extraction-skipped pass — quota to 088 fold)

- Pre-checks live: `wc -l voyage/supervisor.py` → 2624 at pass start
  (unchanged since the tape pass); `git diff --name-only --
  Voyage/voyage/supervisor.py` empty, so the region was quiet.
  Concurrent tracks hold foreign hunks in `voyage/registry_records.py`
  (mid-landing causvid/sfx family facades), `LTX2.md`,
  `issues/031/035/152` — none inside the supervisor region (final
  `git diff` on owned scope shows only the 088 fold files).
- Extraction skipped — no block met the verbatim + zero-cross-ref
  rule (verified via rg/sed before moving anything):
  - Routing (`audio_worker_module` :204 / `video_worker_module` :213
    + `VIDEO/AUDIO_WORKER_MODULES` :123/:138 +
    `STREAMING_VIDEO_BACKENDS` :130) is module-level and movable in
    isolation, but it is issue 023's unification area — verified NOT
    landed (`supervisor.py:130` still hand literal with internal uses
    at :243/:1427/:2006/:2081; `backends.py:82-88` still reads "the
    supervisor track owns unifying this with
    supervisor.STREAMING_VIDEO_BACKENDS"). Untouched per the three
    prior passes' deferral.
  - `_read_lock_holder` (:371) uses zero `self` in its body (verified
    via sed — no `self.` lines), but it is an instance method with 6
    call sites through `self` (5 in
    `tests/test_supervisor_hardening.py:227-235` + internal
    `supervisor.py:340` from `_held_run_lock`): moving it rewrites
    every caller, so it is not verbatim-movable, and the `_held_run_lock`
    call is a cross-reference outside its block.
  - `_stored_relative` (:402) has 2 internal callers (:1656, :2445) —
    cross-references outside its block.
  - Every other method (`_held_run_lock`, `_stage`, `_log_metric`,
    `_pause/_stop_requested`, `_log_rejection`, `_segment_plan_info`,
    `_write_state_preserving_control_plane`, commit/audio/cover
    pipeline) reads `self._run_dir` / `self._log_metric` /
    `self._config` / workers — stateful, not pure helpers.
- Per the brief the pass quota went to ONE 088 fold instead (see 088
  prefetch-shutdown entry): `tests/test_prefetch_shutdown.py` (2
  tests) → `tests/test_generation_stack.py` (18→20 tests), gates.sh
  mypy entry moved in the same edit. No `supervisor.py` edit made.

## Resolution (2026-10-01, extraction-skipped pass)

- Verdict: RECORDED — no extraction this pass (nothing cleanly
  movable); full god-module decomposition remains open.
- Files changed: none under this issue (fold files live under 088).
- DESIGN proposals: "No DESIGN text change proposed: this pass makes
  no supervisor-seam change; the candidate ranking above
  (routing-unification per 023 first, then stateful-helper
  method-groups, never single stateful methods) stands for the next
  quiet-tree pass."
- Residuals: unchanged from the tape pass (proposal/prefetch/
  commit-types/tape landed; commit pipeline methods vs lifecycle vs
  audio-coverage; `sha256_file` shim; streaming-set derivation per
  023/083; worker-module map merge; deterministic-payload compat;
  legacy-migration threading; `run_id` legacy) — each a future
  single-group pass with its own agreement tests.

## Progress log (2026-10-01, extraction-skipped pass — quota to 088 fold)

- Pre-checks live: `wc -l voyage/supervisor.py` → 2666 at pass start
  (+42 vs the 2624 tape-pass as-left — the delta is the foreign
  take-joint compensation hunk, not this track); `git diff --name-only
  -- voyage/supervisor.py` NON-EMPTY (foreign uncommitted hunks: media
  imports `ABSORPTION_EPSILON_SECONDS`/`_take_joint_fade` at :55-58 +
  slice-walk `piece_bounds`/`joint_fade` compensation at :1689-1769),
  so the quiet-region rule FAILS — no supervisor edit attempted.
  Concurrent tracks also hold foreign hunks in
  `tests/test_stage_a_telemetry.py` (read-only, never touched),
  `LTX2.md`, `AGENTS.md`, `DESIGN.md` — all left intact.
- Extraction skipped — no block met the verbatim + zero-cross-ref
  rule (re-verified via rg/sed, same ranking as batch 15):
  - Routing (`audio_worker_module` :206 / `video_worker_module` :215
    + `VIDEO_WORKER_MODULES` :125 / `STREAMING_VIDEO_BACKENDS` :132 /
    `AUDIO_WORKER_MODULES` :140) is module-level and movable in
    isolation, but it is issue 023's unification area — verified NOT
    landed (`supervisor.py:132` still hand literal with internal uses
    at :245/:1429/:2048/:2123; `backends.py:82` still reads "the
    supervisor track owns the rewire, issue 023"). Untouched per all
    prior passes' deferral.
  - `_read_lock_holder` (:373) uses zero `self` in its body (verified
    via sed — no `self.` lines), but it is an instance method with an
    internal caller (`supervisor.py:342` from `_held_run_lock`) + 5
    test call sites (`tests/test_supervisor_hardening.py:227-235`):
    moving it rewrites every caller, so it is not verbatim-movable,
    and the `_held_run_lock` call is a cross-reference outside its
    block.
  - `_stored_relative` (:404) has 2 internal callers (:1658, :2487) —
    cross-references outside its block.
  - `_checked_tape_path` (:417) reads `self._run_dir` (2 `self.`
    uses) + has an internal caller (:2200) — stateful, not a pure
    helper.
  - Every other method (`_held_run_lock`, `_stage`, `_log_metric`,
    `_pause/_stop_requested`, `_log_rejection`, `_segment_plan_info`,
    `_write_state_preserving_control_plane`, commit/audio/cover
    pipeline) reads `self._run_dir` / `self._log_metric` /
    `self._config` / workers — stateful, not pure helpers.
- Per the brief the pass quota went to ONE 088 fold instead (see 088
  final-blend-scale entry): `tests/test_final_blend_scale.py` (4
  tests) → `tests/test_finalize_fastpath.py` (6→10 tests), gates.sh
  mypy entry moved in the same edit. No `supervisor.py` edit made.

## Resolution (2026-10-01, extraction-skipped pass)

- Verdict: RECORDED — no extraction this pass (dirty tree + nothing
  cleanly movable); full god-module decomposition remains open.
- Files changed: none under this issue (fold files live under 088).
- DESIGN proposals: "No DESIGN text change proposed: this pass makes
  no supervisor-seam change; the candidate ranking above
  (routing-unification per 023 first, then stateful-helper
  method-groups, never single stateful methods) stands for the next
  quiet-tree pass."
- Residuals: unchanged from the tape pass (proposal/prefetch/
  commit-types/tape landed; commit pipeline methods vs lifecycle vs
  audio-coverage; `sha256_file` shim; streaming-set derivation per
  023/083; worker-module map merge; deterministic-payload compat;
  legacy-migration threading; `run_id` legacy) — each a future
  single-group pass with its own agreement tests.
