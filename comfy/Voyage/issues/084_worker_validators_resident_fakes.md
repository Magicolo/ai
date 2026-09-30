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
