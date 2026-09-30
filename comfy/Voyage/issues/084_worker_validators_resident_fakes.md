# 084 — Collapse worker duplication: shared validators + resident stack + port fakes onto `video_common`

- Severity: HIGH (structure)
- Files: `voyage/workers/video.py` (200L), `audio.py` (197L), `sfx.py` (145L), `audio_acestep.py` (291L), `sfx_mmaudio.py` (382L), `video_common.py` (342L), `loop.py` (183L)
- Area: workers — dedup
- Decision: Q&A locked — collapse all, not validators-only

## Description

Audio/SFX validators triplicated (`validate_sample_rate` + `validate_channels` identical in `audio.py:55-64`, `audio_acestep.py:58-67`, `sfx_mmaudio.py:55-64`; `validate_output_path` in `video.py:55-58` + `audio.py:67-70` + `sfx.py:34-37`). `_require_torch` ×3 + `_require_module` ×1; lazy resident-stack pattern ×2 (`_require_stack/_models_dir/_device/handle_init/health/evict/shutdown` in acestep vs sfx_mmaudio, structure identical); FLAC→WAV `_convert` ×2 (same ffmpeg arg-list, different error string). Fake `video/audio/sfx` (542L combined) hand-roll serve maps + warmup loops while `video_common.run_benchmark_harness` + `standard_serve_map` already exist for the GPU trio. Fake `sfx.py:17,20` imports validators from GPU modules (reverse dep). GPU trio still repeats health/benchmark/evict/rebuild + `1024**3` formatting + `validate_*` after the `video_common` half-extraction.

## Rationale

~60L triple copies + two full stack lifecycles + three bespoke loops for one contract each. Reverse fake→GPU import makes slim import GPU modules. Every validator fix needs 3 edits.

## Live evidence

- `grep -n "def validate_sample_rate\|def validate_channels\|def validate_output_path\|def _require_torch\|def _convert" voyage/workers/*.py voyage/audio/*.py`
- `voyage/workers/video_common.py:10-13` documents deliberate non-shares (LongLive `.pt`, imageio writer) — the rest is shareable
- `voyage/workers/loop.py` already owns `validate_benchmark_counts` (good precedent)

## Repro

```bash
grep -n "validate_sample_rate\|validate_channels\|validate_output_path\|BYTES_PER_GIB\|1024\*\*3" voyage/workers/*.py voyage/*.py | head -n 40
grep -n "standard_serve_map\|run_benchmark_harness" voyage/workers/*.py
```

## Fix candidates

1. New `voyage/workers/_validators.py` (sample_rate/channels/output_path/duration/fps/frame_count/geometry/energy) — kills triple copies, fixes reverse import.
2. New `voyage/workers/_resident.py` (`find_spec` guard + lazy-init + `_convert` + evict/shutdown; director `_require_module` as template); single `BYTES_PER_GIB` export.
3. Port fake `video/audio/sfx` onto `standard_serve_map` (generalize op param) + `run_benchmark_harness`; collapse `_save_mp4` wrappers (ltxv+causvid) into `video_common.save_mp4` call sites.
4. Gate: `gates.sh` green + `test_audio_workers`, `test_sfx_contract`, `test_video_common`, `test_benchmark_counts` green.

## Refs

- Issue 019 (video_common extraction); `voyage/workers/video_common.py`, `voyage/workers/loop.py`

## Progress log

- 2026-09-30 (this resolution): re-verified every premise live
  (`voyage:latest`, CPU-only, `grep` over `voyage/workers/*.py`):
  `validate_sample_rate` + `validate_channels` identical in `audio.py`,
  `audio_acestep.py`, `sfx_mmaudio.py`; `validate_output_path` in
  `video.py` + `audio.py` + `sfx.py`; `_require_torch` x3 (video fakes
  have none — GPU ban holds trivially) + `BYTES_PER_GIB = 1024**3` x2
  (acestep/sfx_mmaudio); resident-stack shape x2 (`_require_stack` /
  `_models_dir` / `_device` / `handle_init` / `health` / `evict` /
  `shutdown` mirrored); FLAC→WAV `_convert` x2 (same ffmpeg arg-list
  shape, different flags/messages); fake `sfx.py:20` reverse-imports the
  GPU worker (`from voyage.workers.sfx_mmaudio import ...`); fake
  `video/audio/sfx` hand-roll serve maps + warmup loops while
  `video_common.standard_serve_map` + `run_benchmark_harness` serve the
  GPU trio. `validate_duration_seconds` is NOT triplicated identically:
  audio/acestep take finite->0 while MMAudio adds a
  `MAX_WINDOW_SECONDS` ceiling — base + specialization, not three copies.
- TDD failing-first (`tests/test_worker_validators_unified_084.py`,
  4 tests): watched `ImportError: cannot import name '_validators'`
  in-container, then green.
- Implemented: new `voyage/workers/_validators.py` (9 validators:
  sample_rate/channels/output_path/geometry/fps/frame_count/energy/
  window_id/duration-base + `MIN/MAX_ENERGY`) and new
  `voyage/workers/_resident.py` (`BYTES_PER_GIB` single export +
  `require_module`/`require_torch` `find_spec` guards, director
  `_require_module` as template); rewired `video.py`/`audio.py`/`sfx.py`/
  `audio_acestep.py`/`sfx_mmaudio.py` onto them with compat re-exports
  (existing `audio_acestep.validate_*` / `audio_worker.validate_*`
  imports keep working — pinned by `test_audio_request_validation.py` +
  `test_audio_workers.py`); reverse import gone (`sfx.py` imports
  `_validators`, duration stays from `voyage.audio.mmaudio_sfx` which is
  the CPU-safe lib, not a worker); fake `video` onto
  `standard_serve_map("video", ...)` (preserves the historical
  `"video-<segment>"` checkpoint prefix — `backend_name="fake"` would
  relabel to `"fake-..."`) with dead `handle_checkpoint`/`handle_shutdown`
  deleted (map-owned now); fake `video` benchmark onto
  `run_benchmark_harness` (response keys unchanged, seed-varying probe
  preserves the original per-block hue behavior). Hot files
  (`supervisor.py`/`director.py`/`video_ltxv.py`) untouched throughout.
- Lesson (do not regress): `ruff check --fix` deletes bare re-export
  imports (F401) — use the `from m import x as x` self-alias idiom for
  intentional re-exports (caught once on `media.py`, same class applies
  to any future `_validators` pure re-export).
- Gates (touched files only): `ruff check` + `format --check` + `mypy
  strict` clean on `_validators.py`/`_resident.py`/`video.py`/`audio.py`/
  `sfx.py`/`audio_acestep.py`/`sfx_mmaudio.py`/new test; suites green:
  new 4 + `test_audio_workers` + `test_sfx_contract` + `test_video_common`
  + `test_benchmark_counts` + `test_benchmark` (48 passed).

## Resolution

- Verdict: FIXED (safe subset; audio/sfx serve-map generalization open).
- Files changed: `voyage/workers/_validators.py` (new),
  `voyage/workers/_resident.py` (new), `voyage/workers/video.py`
  (shared imports + harness benchmark + standard serve map),
  `voyage/workers/audio.py` (shared imports, `MIN/MAX_ENERGY` alias),
  `voyage/workers/sfx.py` (reverse import removed),
  `voyage/workers/audio_acestep.py` + `voyage/workers/sfx_mmaudio.py`
  (shared `BYTES_PER_GIB`/guards/validators, compat names kept),
  `tests/test_worker_validators_unified_084.py` (new, 4 tests).
- Residual (precise): audio/sfx serve-map generalization (needs a
  `video_common` op-name parameter — touches the hot GPU workers'
  `standard_serve_map` call sites, deferred to a quiet tree); full
  resident-stack collapse (`_require_stack`/`_convert`/evict differ by
  stack type, init signature, and convert flags/messages);
  `_save_mp4` wrapper collapse (ltxv+causvid — touches hot
  `video_ltxv.py`); audio/sfx benchmark harness ports (mechanical,
  same pattern as the video port this pass).
