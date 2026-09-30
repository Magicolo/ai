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
