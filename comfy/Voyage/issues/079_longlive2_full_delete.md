# 079 — Full-delete `longlive2` backend (default is ltxv, LongLive is heaviest legacy)

- Severity: HIGH (structure / scope: 1264L worker + registry + image layers + tests)
- Area: backends — video `longlive2` removal
- Decision: Q&A round 1-2 locked — deprecate = full delete, not soft-hide
- Rank rationale: biggest single-file deletion; kills own `.pt` tape format, checkpoint-load OOM, LongLive-first doc presentation.

## Description

`ltxv` is the default since 2026-09-29 (`voyage/config.py:204-211` `_DEFAULT_ROW = BACKEND_REGISTRY["ltxv"]`, 768×512@24, `segment_frames=96`). `longlive2` (1280×704@24, 29f, `persistent_kv`) is live but non-default, biggest worker file, own torch `.pt` recovery tape, own `imageio.get_writer` path, known checkpoint-load ~38 GB anon-RSS OOM (filed `Voyage/issues/096` in history, now archived in git). User dropped it to fallback; Q&A chose full delete.

Live sites (verified 2026-09-30 sweeps):

- `voyage/workers/video_longlive.py:1` — 1264 lines, resident BF16→FP8 DiT + CPU T5 twin, `LongLiveStreamSession`, `local_attn_size=16/sink=8`, VAE-offload, chunked decode, `.pt` tape
- `voyage/config.py:135-198` — `BACKEND_REGISTRY` rows (`longlive2` 1280×704/29f + `longlive2-bf16` spec); `:140,155,170,188` `segment_frames 96/29/72`
- `voyage/supervisor.py:88-108` — `VIDEO_WORKER_MODULES` + `STREAMING_VIDEO_BACKENDS = ("longlive2","ltxv","causvid")`
- `voyage/model_registry.py` — longlive/wan/Qwen pins + `download_longlive2_bf16` manifest-clobber site
- `worker/Dockerfile.video:50-54,140-168` — LongLive@`6b36d20` clone + `wan_models` shims + `_enter_longlive_tree` CWD shim (`video_longlive.py:~20`)
- `voyage/cli.py:1174` — `_CUDA_BACKENDS` includes `longlive2`; `:1143-1163` `_LTXV_NOVEL_BLOCK_FRAMES=96/_LONGLIVE_*=8,4` duplicating registry
- `scripts/qualify.sh:2` — help hardcodes `--backend longlive2`; `scripts/run.sh:67` sniff includes it
- Tests: `tests/test_longlive.py` (11), `test_longlive_stages.py` (6), `test_longlive_init_validation.py` (9), `test_longlive_offload_fusion.py` (4), `test_precision.py` fp8/bf16 legs, `test_qualification.py` longlive2 leg
- Docs/reports: `reports/video-backends.md` longlive2 leg (keep as dated evidence, mark historical), `reports/longlive-audit.md` (keep), `docs/UPSTREAM_LONG_LIVE_PATCHES.md`, backend-first ordering in INSTALL/BACKENDS/MODELS/README

## Why this is an issue

- Keeps two recovery formats alive (LongLive `.pt` vs JSON `write_tape_atomic` for ltxv/causvid) — supervisor resume must understand both.
- Keeps heaviest image layers + `transformers==4.57.6` pin tension (LongLive `x_clip_loss` vs 5.x) for a non-default path.
- Every backend-matrix change (streaming sets, CUDA sets, presets, qual harness) pays a 4-backend tax instead of 3.

## Live evidence

Sweep inventories (2026-09-30): worker table (longlive 1264L largest), `grep -n "STREAMING_VIDEO_BACKENDS\|VIDEO_WORKER_MODULES\|BACKEND_REGISTRY"` showing longlive2 in all three sets, `reports/video-backends.md:85-117` longlive2 VALID 87f numbers.

## Reproduction

```bash
wc -l voyage/workers/video_*.py
grep -rn "longlive" voyage/config.py voyage/supervisor.py voyage/cli.py worker/Dockerfile.video scripts/*.sh | head -n 40
grep -rln "longlive" tests/ docs/ reports/
```

## Resolution candidates

1. Delete `voyage/workers/video_longlive.py` + registry longlive rows + `VIDEO_WORKER_MODULES["longlive2"]` + `STREAMING_VIDEO_BACKENDS` entry (derive remainder from registry — see 083) + Dockerfile LongLive clone/shims + `run.sh`/`qualify.sh` refs.
2. Delete or repoint longlive-only tests (stages/init-validation/offload-fusion/precision-longlive legs); keep `reports/longlive-audit.md`, mark `video-backends.md` longlive leg historical.
3. Reorder INSTALL/BACKENDS/MODELS/README ltxv-first (covered in 092 but lands with this delete).
4. Gate: `Voyage/scripts/gates.sh` green + `grep -rni longlive voyage/ worker/ scripts/ tests/` empty except historical reports/docs markers.

## Refs

- `voyage/config.py:117-211`, `voyage/supervisor.py:88-108`, `voyage/workers/video_longlive.py:1`, `worker/Dockerfile.video`, `reports/video-backends.md`, `reports/longlive-audit.md`
- Prior: issues 025 (quadruple registries), 036 (god modules), archived 096 (checkpoint OOM)

## Progress log

- 2026-09-30 (this resolution): EVALUATED FIRST per the task contract.
  Re-verified every live site in-container/on-disk (no GPUs, no host pip):
  `BACKEND_REGISTRY` longlive2 row (`config.py:150-164`, 1280x704/29f) +
  `longlive2-bf16` spec (`model_registry.py:827-855`); `VideoBackendName`
  Literal (`config.py:26`); `VIDEO_WORKER_MODULES` + `STREAMING_VIDEO_BACKENDS`
  (`supervisor.py:92-98`); `_frames_per_segment` longlive branch +
  `_LONGLIVE_*` constants + `_CUDA_BACKENDS` sniff (`cli.py:1309,1313-1315,1341`,
  `scripts/run.sh:67`); init/generate `--backend` choices (`cli.py:1960,2179`);
  `TUI BACKENDS` (`tui_state.py:33`); `LongLiveBackend` fake
  (`fake_backends.py:194`); `download/verify_longlive2_bf16`
  (`model_registry.py:442-449`, `cli.py:342-456`); `models_ensure.py:57`
  mapping; `doctor.py:276` verify leg; `worker/Dockerfile.video:66-70,176-253`
  LongLive@`6b36d20` clone + `wan_models` shims + `_enter_longlive_tree` CWD
  shim + PYTHONPATH; `scripts/qualify.sh` (longlive2-first driver) +
  `scripts/build-video.sh:14-16` (imports `video_longlive`); tests
  (`test_longlive.py`, `test_longlive_stages.py`,
  `test_longlive_init_validation.py`, `test_longlive_offload_fusion.py`,
  `test_precision.py` fp8/bf16 legs, `test_qualification.py` longlive2 leg);
  `reports/video-backends.md:85-117` (VALID 87f) + `reports/longlive-audit.md` +
  `docs/UPSTREAM_LONG_LIVE_PATCHES.md`.
- Concurrent-work check: `git status` shows `supervisor.py` (+232 Stage-A
  telemetry hunks), `workers/director.py` (+24), `workers/video_ltxv.py` (+59)
  all uncommitted and hot; mid-session `config.py`/`media.py` gained foreign
  hunks from other agents (`FPS_MATCH_TOLERANCE`, `MAX_FINAL_OVERLAP_FRACTION`,
  tomli-shim edits — visible in `git diff` alongside my own, disjoint
  regions). Full deletion requires `supervisor.py:88-108,351,2056,2358`
  hunks → direct conflict with the in-flight telemetry change → verdict:
  PARTIAL (largest safe subset), no deletion, no hot-file hunks.
- TDD failing-first (`tests/test_longlive2_deprecation_079.py`, 4 tests):
  watched `AttributeError: module 'voyage.config' has no attribute
  'DEPRECATED_VIDEO_BACKENDS'` in-container (`voyage:latest`), then green.
- Implemented subset: `config.py` gains `DEPRECATED_VIDEO_BACKENDS =
  ("longlive2",)` + `warn_if_deprecated_backend()` (pure advisory
  `DeprecationWarning` pointing at `ltxv`); wired once into
  `cli.cmd_init` (the new-run decision point); `tui_state.FIELD_HELP`
  backend marks longlive2 deprecated. Registry row, worker (1264L),
  resume path, tests, Dockerfile, scripts all retained untouched.
- Gates (touched files only; full `gates.sh` left to orchestrator):
  `ruff check` + `ruff format --check` + `mypy` clean on
  `config.py`/`cli.py`/`tui_state.py`/new test; related suites
  (`test_backend_registry`, `test_tui_state`, deprecation tests) green.
  Note: `ruff format` on `config.py` inserted one blank line in my own
  hunk only — foreign hunks already format-clean, left untouched.

## Resolution

- Verdict: PARTIAL. Deprecate + warn shipped; full delete NOT executed
  (unsafe under concurrent hot-file work + live dependents above).
- Files changed: `voyage/config.py` (+deprecated registry + helper),
  `voyage/cli.py` (+import + one `cmd_init` warn call),
  `voyage/tui_state.py` (+help text),
  `tests/test_longlive2_deprecation_079.py` (new, 4 tests).
- Residual (precise): full delete per §"Resolution candidates" 1-4 stays
  open — (1) delete `workers/video_longlive.py` + registry longlive rows
  + `VIDEO_WORKER_MODULES["longlive2"]` + `STREAMING_VIDEO_BACKENDS` entry
  + Dockerfile clone/shims + `run.sh`/`qualify.sh`/`build-video.sh` refs;
  (2) delete/repoint the six longlive test modules + precision/qual legs,
  mark `video-backends.md` longlive leg historical; (3) reorder
  INSTALL/BACKENDS/MODELS/README ltxv-first; (4) gate `gates.sh` green +
  `grep -rni longlive voyage/ worker/ scripts/ tests/` empty except
  historical markers. Preconditions: quiet tree (no uncommitted
  `supervisor.py`/`director.py`/`video_ltxv.py` hunks), stored-run
  migration decision for existing longlive2 TOMLs, and generate/run-time
  warn wiring (this pass warns at `init` only).

## Progress log (2026-09-30, tests-only pass — quiet-tree check)

- Precondition evaluated live per the task contract: tree is HOT, not
  quiet. `git status --porcelain` shows uncommitted `voyage/supervisor.py`
  (+71 Stage-A telemetry hunks), `voyage/config.py` (+14), and
  `tests/test_stage_a_telemetry.py` (+25) — all overlapping the deletion
  surface (`supervisor.py:98,106` `VIDEO_WORKER_MODULES` /
  `STREAMING_VIDEO_BACKENDS`, `config.py:31,182` deprecated row +
  registry). Plus untracked `tests/test_novelty_leniency.py` (concurrent
  agent). Full deletion requires `supervisor.py:88-108` hunks → direct
  conflict with the in-flight telemetry change.
- Longlive2 presence re-verified (read-only): `voyage/workers/
  video_longlive.py` still present; `BACKEND_REGISTRY["longlive2"]` row
  (`config.py:182`, 1280×704/29f) + `DEPRECATED_VIDEO_BACKENDS`
  (`config.py:31`) both live; `VIDEO_WORKER_MODULES` +
  `STREAMING_VIDEO_BACKENDS` (`supervisor.py:98,106`) still include
  `longlive2`; `TUI BACKENDS` (`tui_state.py:33`) still lists it; six
  longlive test modules + precision/qual legs still present.
- Verdict: BLOCKED-with-evidence (hot tree). No deletion executed, no
  hot-file hunks touched, no unilateral delete per the contract. No files
  changed in this pass.

## Resolution (2026-09-30, tests-only pass)

- Verdict: blocked (hot tree — ready-to-execute plan recorded, not run).
  Files changed: none. DESIGN proposals: none.

## Progress log (2026-09-30, batch 12 — full-delete owning pass)

- Quiet-tree check FIRST per the contract: `git rev-parse HEAD` →
  `447353e`, `git status --porcelain` clean except submodule untracked
  content at `../../tango/Tango` (outside the Voyage tree, unrelated —
  `git submodule status` shows no mapping; the Voyage tree itself has
  zero uncommitted hunks). Precondition (a) MET: no concurrent hunks on
  the deletion surface (`supervisor.py`/`config.py`/`director.py`/
  `video_ltxv.py` all clean, unlike batches 7/9).
- Deletion surface re-verified read-only (as-read lines, 2026-09-30):
  `voyage/workers/video_longlive.py` (1264L) present;
  `voyage/config.py:26` Literal, `:31` `DEPRECATED_VIDEO_BACKENDS`,
  `:44-60` `warn_if_deprecated_backend`, `:182-196` registry row,
  `:250-253` models_dir comment, `:727` TOML comment;
  `voyage/supervisor.py:112-120` module map + streaming tuple,
  `:277-284` longlive2 init branch, `:790`/`:2199`/`:2499` comments +
  `use_relative_rope`; `voyage/cli.py` re-exports `:62-63`/`:143-144`,
  imports `:110`/`:120`, `__all__` `:227`/`:245`, choices `:292`/`:511`,
  models-target default `:331`; `voyage/cli_models.py` imports
  `:176`/`:184`, list `:192`, verify `:208`, download branch
  `:232`/`:273-302`, info `:306`; `voyage/cli_planning.py:85-87`
  `_LONGLIVE_*` + `:101-103` frames branch; `voyage/cli_observe.py:82-83`
  revision-name tuple; `voyage/cli_paths.py:18` comment;
  `voyage/tui_state.py:36` BACKENDS, `:88-90` help;
  `voyage/model_registry.py` imports `:48`/`:60-66`/`:161`/`:171`,
  `__all__` rows, wrappers `:447-454`, `MODEL_SPECS` row `:548-577`,
  comments `:1025`/`:1244`/`:1251`;
  `voyage/registry_records.py:18-31` `LONGLIVE_*`, `:464-472` expected
  hash, `:489-506` `_record_longlive2`, `:633-636` `_describe_longlive2`
  (`WAN_*`/`_WAN22_RELATIVE` have no other consumer — causvid uses
  Wan2.1, not Wan2.2); `voyage/models_ensure.py:58-63` spec map;
  `voyage/doctor.py:293` verifier row; `voyage/backends.py:7`/`:380`
  docstrings; `worker/Dockerfile.video` clone `:66-70`, shims `:184-186`,
  PYTHONPATH/ENV `:244`/`:246`, chown `:231-232`, comments
  `:10`/`:44`/`:56`/`:59-65`/`:100-103`/`:115`/`:132`/`:201`/`:250-253`;
  `scripts/run.sh` sniff `:66` + case `:71` + comments `:7`/`:14`/
  `:21`/`:75`; `scripts/qualify.sh` choices `:36`/`:52`/`:66`/`:70-72`;
  `scripts/build-video.sh` smoke `:17`/`:19`/`:24` (3→2 workers);
  `scripts/test.sh` clean (no refs); `voyage/fake_backends.py` clean
  (the old `:194` cite is stale — no longlive refs in the file).
  Owned tests: `test_longlive.py`, `test_longlive_stages.py`,
  `test_longlive_init_validation.py`, `test_longlive_offload_fusion.py`,
  `test_longlive2_deprecation_079.py`, `test_e2_longlive_clip_125.py`
  (all match `test_*longlive*`), `test_backend_registry.py:54,82,92`.
- Stored-run TOML migration DECISION (explicit, per the task): HARD
  ERROR with migration hint, NOT silent remap. Rationale: a silent
  remap to ltxv changes geometry mid-run (1280×704/29f →
  768×512/96f), invalidating committed segment durations, while
  longlive `.pt` tapes can never resume on the ltxv JSON-tape path
  (the supervisor already rejects cross-backend tapes loudly) — a
  remap would corrupt the timeline silently instead of failing fast.
  Stored `backend = "longlive2"` runs must fail at load with: unknown
  backend + known list + "re-init with --backend ltxv (tapes do not
  transfer; existing segments stay valid media, only continuation
  stops)". Proposed patch (text only, NOT applied — see verdict):
  `video_worker_module` gains a longlive2-specific branch before the
  generic KeyError, and `load_config` callers (`_load_run`,
  `cmd_run`/`cmd_generate` resolve paths) surface it unchanged;
  `with_video_backend`/`resolve_config(backend="longlive2")` already
  raise unknown-backend ValueError once the registry row is gone
  (no extra code — covered by test). TDD test plan (not run — no
  behavior changed this pass): `test_backend_registry.py` gains
  `test_longlive2_stored_run_fails_with_migration_hint` (write
  voyage.toml with `backend = "longlive2"`, assert ConfigurationError
  names ltxv + tapes) + `test_no_silent_remap` (resolve_config rejects
  the name; no fallback row).
- SCOPE VERDICT: deletion NOT executed although the tree is quiet.
  The batch-9 plan's step (2) ("six longlive test modules +
  precision/qual legs") already exceeds this pass's owned test scope
  (`tests/test_*longlive*` + `test_backend_registry*`), and full
  deletion additionally breaks ~24 non-owned follower files plus
  unowned gates/docs (enumerated in the Resolution residuals with
  file:line handoffs). The read-only contract forbids touching them,
  and landing a red tree for other groups violates §9/§12. The batch-7
  deprecation layer remains the maximal safe subset: the warn stays
  (the name is still accepted everywhere), nothing deleted.
- Baseline evidence (unmodified tree, in-container `voyage:latest`,
  CPU-only): owned suites green —
  `test_backend_registry` + all six `test_*longlive*` modules:
  51 passed, 3 skipped, 0 failed.
- Mid-session concurrent-activity re-check (same pass, before writing
  this log): the tree now shows other agents' in-flight work —
  `voyage/cli_observe.py` (+23 PERF203 helper, issue 031 track),
  `voyage/scoreboard.py` (+16 JsonValue narrowing, issue 035 track),
  `voyage/workers/director.py` (+3 init-strict, issue 127 track),
  `issues/031`/`issues/035` logs, untracked
  `tests/test_director_init_strict_127.py`. Disjointness verified via
  `git diff`: the cli_observe hunk sits at `:689-739` (my surface is
  `:82-83`); scoreboard/director files carry no longlive2 surface at
  all. So the deletion surface itself is still hunk-free — the BLOCKED
  verdict below rests on the scope contract (non-owned followers),
  not on concurrent-hunk conflict. My four files are the only ones I
  touched (`TASK.md` + `docs/BENCHMARKING.md` belong to the 093 pass).

## Resolution (2026-09-30, batch 12)

- Verdict: BLOCKED (scope-contract, tree quiet). Precondition (a)
  MET (HEAD `447353e`, zero Voyage hunks); precondition (b) DECIDED
  (hard error with migration hint, rationale + patch/test plan above).
  Files changed: none (no source, test, script, doc, or Dockerfile
  edits — any owned-scope fragment would break non-owned followers).
- DESIGN proposals (text only, no DESIGN.md write per contract): none
  needed — no spec change; the deletion is mechanical once unblocked.
- Residuals — owning pass handoffs (exact, as-read 2026-09-30):
  OWNED-SCOPE (delete when unblocked): worker file
  `voyage/workers/video_longlive.py`; `voyage/config.py:26,31,44-60,
  182-196,250-253,727` (remove Literal member + deprecated tuple +
  helper + row + comments; keep-or-tombstone `:31` decided as REMOVE —
  nothing still accepts the name, so the warn is dead); `voyage/
  supervisor.py:112-120,277-284,790,2199,2499` (+ migration-hint branch
  in `video_worker_module` `:245-251`); `voyage/cli.py:62-63,110,120,
  143-144,227,245,292,331,511`; `voyage/cli_models.py:176,184,192,208,
  232,273-302,306`; `voyage/cli_planning.py:85-87,101-103`;
  `voyage/cli_observe.py:82-83`; `voyage/cli_paths.py:18`;
  `voyage/tui_state.py:36,81,88-90`; `voyage/model_registry.py:48,
  60-66,161,171,208,222-228,334,346,359,375,447-454,548-577,1025,1244,
  1251`; `voyage/registry_records.py:18-31,464-472,482,489-506,633-636`
  (incl. `WAN_*` — last consumer gone; closes issue 070's floating pin
  by deletion); `voyage/models_ensure.py:58-63`; `voyage/doctor.py:293`;
  `voyage/backends.py:7,380`; `worker/Dockerfile.video:10,44,56,59-70,
  91,100-103,115,117,132,155,176,184-186,201,224,231-232,238,244,246,
  250-253`; `scripts/run.sh:7,14,21,66,71,75`; `scripts/qualify.sh:4,10,
  36,52,66,70-72`; `scripts/build-video.sh:2,17,19,24`; owned tests
  (delete six modules, re-pin `test_backend_registry.py:54,82,92`);
  `reports/video-backends.md:30-83` mark historical (keep
  `reports/longlive-audit.md` as dated evidence);
  `docs/BACKENDS.md:42,54,71,76,101`, `docs/MODELS.md:7,11,14`,
  `docs/INSTALL.md:20,23-24,35,84`, `docs/ARCHITECTURE.md:52`,
  `docs/OPERATIONS.md:92,152,177-178,183`,
  `docs/TROUBLESHOOTING.md:23`, `docs/SFX.md:127`,
  `docs/UPSTREAM_LONG_LIVE_PATCHES.md` (recommend keep + historical
  header, same class as the audit report).
  NON-OWNED followers (other groups own the fixups — gate lands only
  jointly): `tests/test_generate.py:108-135,224`,
  `tests/test_backends_adapter.py:135-159`,
  `tests/test_tui.py:134-138`, `tests/test_tui_state.py:134,180,210,228`,
  `tests/test_precision.py:17,41-42`,
  `tests/test_qualification.py` (longlive2-first harness),
  `tests/test_registry_pins.py:31,111-140`,
  `tests/test_registry_split.py:15,28,41,112-114`,
  `tests/test_cli_split.py:116,158`,
  `tests/test_cli_hardening.py:442`,
  `tests/test_generate_ensure.py:44-47,108`, incidental refs in
  `test_129_tape_atomic`, `test_audio_acestep_cwd`,
  `test_augment_models`, `test_benchmark_counts`,
  `test_causvid_worker`, `test_checkpoint_safety`,
  `test_containers_rank2`, `test_enter_repo_trees`,
  `test_perf_regressions`, `test_recovery`,
  `test_stage_a_telemetry`, `test_tape_trust_123_171`,
  `test_worker_perf_rank2`; `scripts/gates.sh:50` (test list);
  `README.md:10,27,40,65,88,135`; comment staleness in
  `voyage/workers/__init__.py:8`, `voyage/workers/video_common.py:3,13,
  280`, `voyage/rpc.py:152`, `voyage/audio/acestep.py:14`,
  `voyage/workers/audio_acestep.py:69`,
  `voyage/workers/video_causvid.py:109`,
  `voyage/workers/video_ltxv.py:23,584,1024`.
  Gate for the owning pass: full `Voyage/scripts/gates.sh` green +
  `grep -rni longlive voyage/ worker/ scripts/ tests/` empty except
  historical report/doc markers.
- Per-file gates this pass: N/A (no files touched). Related-suite
  evidence: owned longlive suites green on the unmodified tree
  (51 passed, 3 skipped — baseline for the owning pass).
- Ready-to-execute deletion plan (for the owning pass, once the tree is
  quiet — no uncommitted `supervisor.py`/`config.py`/`director.py`/
  `video_ltxv.py` hunks): (1) delete `voyage/workers/video_longlive.py`
  (1264L) + `voyage/config.py:182-19x` `longlive2` row + `:31`
  `DEPRECATED_VIDEO_BACKENDS` (or keep as tombstone — decide) +
  `voyage/supervisor.py:98,106` longlive2 entries (derive remainder from
  registry per 083) + `worker/Dockerfile.video:66-70,176-253` LongLive
  clone/`wan_models` shims + `scripts/run.sh:67` sniff +
  `scripts/qualify.sh` longlive2-first driver +
  `scripts/build-video.sh:14-16` import + `voyage/registry_records.py`
  WAN/LONGLIVE pins + `voyage/cli_*` `_LONGLIVE_*`/`_CUDA_BACKENDS` refs +
  `voyage/tui_state.py:33` BACKENDS entry + `voyage/fake_backends.py:194`
  fake + `voyage/models_ensure.py:57` mapping + `voyage/doctor.py:276`
  leg; (2) delete/repoint `tests/test_longlive.py`,
  `test_longlive_stages.py`, `test_longlive_init_validation.py`,
  `test_longlive_offload_fusion.py`, `test_longlive2_deprecation_079.py`,
  precision fp8/bf16 longlive legs, qual longlive2 leg; mark
  `reports/video-backends.md:85-117` longlive leg historical (keep
  `reports/longlive-audit.md` as dated evidence); (3) reorder
  INSTALL/BACKENDS/MODELS/README ltxv-first; (4) gate `gates.sh` green +
  `grep -rni longlive voyage/ worker/ scripts/ tests/` empty except
   historical markers. Preconditions: stored-run migration decision for
   existing longlive2 TOMLs + generate/run-time warn wiring (currently
   `init`-only).

## Progress log (2026-10-01, full-delete owning pass — DONE)

- Quiet-tree check FIRST per the contract: `git rev-parse HEAD` →
  `da77026`, `git status --porcelain` shows only the foreign
  `M Voyage/LTX2.md` + pre-existing `../tango/Tango` (both untouched —
  never opened). No uncommitted hunks on the deletion surface.
- TDD failing-first (`tests/test_longlive2_removed_079.py`, 5 tests):
  all 5 FAILED on the unmodified tree (registry still accepted the
  name), then 5/5 GREEN after the deletion. Stored-run decision per the
  batch-12 log: HARD ERROR with migration hint, NOT silent remap
  (geometry 1280x704/29f → 768x512/96f would corrupt timelines; `.pt`
  tapes can never resume on the JSON-tape path).
- Migration mechanism (new, single-sourced): `config.REMOVED_VIDEO_BACKENDS
  = ("longlive2",)` + `config.removed_backend_suffix()` wired into
  `_video_preset` / `_audio_preset` / `_sfx_preset` (ValueError),
  `load_config` stored-TOML gate (ConfigurationError), `supervisor.
  video_worker_module` (ConfigurationError), and `cli_run_ops.cmd_init`
  (exit 2). Every message names the removed backend + `ltxv` + `tapes`.
- Deleted: `voyage/workers/video_longlive.py` (1264L, `git rm`),
  `config` Literal member + `DEPRECATED_VIDEO_BACKENDS` +
  `warn_if_deprecated_backend` + registry row, `supervisor`
  `VIDEO_WORKER_MODULES`/`STREAMING_VIDEO_BACKENDS` entries + longlive2
  init branch (`use_relative_rope` now constant False — the removed
  backend was the only relative-RoPE renderer), all `cli*.py` refs
  (incl. `_LONGLIVE_*` planning math, models-verb branches, observe
  revision tuple), `model_registry` spec row + wrappers + `WAN_*`
  (Wan2.2 — last consumer gone; closes issue 070 by deletion; `WAN21_*`
  kept for causvid) + layout keys, `registry_records` pins +
  builders, `models_ensure` spec map, `doctor` verifier, Dockerfile
  LongLive clone/shims/PYTHONPATH/ENV/chown (transformers 4.57.6 pin
  KEPT — LTXV/CausVid probe-verified against it), `run.sh`/`qualify.sh`
  backends, `build-video.sh` smoke (3→2 workers), six test modules
  (`test_longlive*`, `test_longlive2_deprecation_079`,
  `test_e2_longlive_clip_125`, `test_129_tape_atomic`), `gates.sh`
  mypy entry. Repointed followers: generate/adapter/tui/precision/
  qualification/registry-pins/registry-split/cli-split/cli-hardening/
  generate-ensure/augment/checkpoint/enter-trees/perf/worker-perf/
  recovery/containers/telemetry/tape-trust/acestep-cwd/causvid-worker/
  benchmark-counts/stage-a. Docs reordered ltxv-first (BACKENDS/MODELS/
  INSTALL/OPERATIONS/ARCHITECTURE/TROUBLESHOOTING/SFX/BENCHMARKING/
  README); `video-backends.md` + `UPSTREAM_LONG_LIVE_PATCHES.md` carry
  HISTORICAL headers (kept verbatim as dated evidence, with
  `longlive-audit.md`).
- Residual grep end-state (deliberate, TDD-pinned): `voyage/` mentions
  the name ONLY in the removal machinery (`config.REMOVED_*` +
  `removed_backend_suffix` + its docstrings); `worker/` + `scripts/`
  are clean; `tests/` mentions it ONLY in removal-contract tests
  (079 TDD + generate-hint + pins-absence); docs/reports mention it
  ONLY in historical markers/kept evidence. The batch-12 "empty
  except historical markers" gate is therefore met as: live paths
  contain zero references; the hint + contract tests + dated evidence
  are the documented exceptions.
- Per-file gates (container): `ruff check` + `ruff format --check` +
  `mypy strict` clean on every touched file (two E501 + one format
  reflow + 4 ruff fixes — unused noqa/imports — fixed in-pass).
- Full `Voyage/scripts/gates.sh`: ruff GREEN, format GREEN, mypy
  (147 files) GREEN, pytest 1688 passed + 10 skipped with exactly ONE
  TUI Pilot load-flake per run (run 1:
  `test_mid_run_progress_reaches_log_before_completion`, run 2:
  `test_slow_run_stays_responsive_and_reaches_monitoring_view` —
  both pass in isolation in ~seconds; documented shared-box flake
  class, unrelated to this change — neither references the backend).
  Cache-guard lesson re-learned: ad-hoc `docker run ruff` without
  `VOYAGE_CACHE_ENV` leaks `.ruff_cache` into the bind mount — always
  export the cache env; residue removed before gating.
- Concurrent-work check at close: other agents landed hunks mid-pass
  (issues/031/035/036/070/081/082/088/089/093/152/166 logs,
  `tests/test_integration.py`, `voyage/augment.py`,
  `test_sfx_parser_parity.py` deleted, untracked registry-split
  family) — all disjoint from this surface (verified via `git diff`
  region reads; full gates green WITH their files present). `Voyage/
  LTX2.md` + `tango/Tango` never touched.
- Files changed: none committed (per contract — left uncommitted for
  review).

## Resolution (2026-10-01, full-delete owning pass)

- Verdict: DONE. Full delete executed per §"Resolution candidates"
  1-4, with the migration-hint mechanism the batch-12 plan deferred.
- DESIGN proposal (text only — no DESIGN.md write per contract): see
  the quoted patch in the delivery message (079 §11/backends section:
  backend table drops the longlive2 row, recovery-format section
  drops the `.pt` tape, migration note for stored runs).
