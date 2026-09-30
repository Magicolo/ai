# 085 — Single-source collapse: presets, streaming, maps, escapers, floors, geometry

- Severity: MEDIUM (structure — agreement without enforcement)
- Files: `voyage/config.py:117-202,204,376-390,533-545,571-587,600-609,661-665`, `voyage/cli.py:1095-1118,1123,1761-1767,911-927`, `voyage/tui_state.py:31-41,83-85,112-113,312-354,418-437`, `voyage/supervisor.py:88-108`, `voyage/backends.py:74-89`, `voyage/sfx_finalize.py:54-58`, `voyage/media.py:670-678,802-810,911-913`
- Area: config — single source of truth

## Description

`BACKEND_REGISTRY` (4 rows) is declared the source, but consumers restate it: `_VIDEO/_AUDIO/_SFX_BACKEND_PRESETS` + `_video/_audio/_sfx_preset()` + `with_video_backend` (`config.py:933`)/`apply_draft_overrides` (`config.py:827`, both "new code should call resolve_config" wrappers); `cli._LTXV_NOVEL_BLOCK_FRAMES=96/_LONGLIVE_*/_CAUSVID_NOVEL=72` duplicating `segment_frames`; `tui_state._FALLBACK_FRAMES_PER_SEGMENT=48/_FALLBACK_FPS=24` duplicating the fake row; `default_config_toml:595-629` fallback literals duplicating `_DEFAULT_ROW`. Streaming triplicated (`supervisor.STREAMING_VIDEO_BACKENDS` tuple vs `backends._STREAMING_BACKENDS` frozenset vs `BACKEND_STATE_MODES`); worker-module maps ×3 with 3 near-identical errors; TOML escapers ×2 (`config.py:571-587` vs `tui_state.py:418-437`, TUI adds C0 loop); augment floors ×5 (`AugmentConfig`, `FinalizeOptions`, `AUGMENT_DEFAULT_*`, `GenerateFormState` + help, `[augment]` TOML, CLI help) + `PRESENTATION_MIN_FPS=24` vs 32fps floor dual; geometry/latent literals (`[1,8,48,44,80]` ×5, draft ×2, causvid `[1,21,16,60,104]` ×2); `SPEC_MIN_FREE_SPACE_GIB=20` vs `DEV_MIN_FREE_SPACE_GIB=5` two defaults.

## Rationale

Every backend/floor/geometry change needs N hand edits; one set already diverged once (021 `_CUDA_BACKENDS`). Import-time drift fails on GPU boxes at midnight instead of at the gate.

## Live evidence

- `grep -n "STREAMING_VIDEO_BACKENDS\|_STREAMING_BACKENDS\|VIDEO_WORKER_MODULES\|BACKEND_REGISTRY\|_CUDA_BACKENDS" voyage/*.py`
- `grep -n "interpolated_frame_count\|PRESENTATION_MIN_FPS\|AUGMENT_DEFAULT\|FALLBACK_FRAMES" voyage/*.py voyage/tui_state.py`
- `grep -n "1,8,48,44,80\|1,8,48,22,40\|1,21,16,60,104" voyage/config.py voyage/workers/video_causvid.py`

## Repro

```bash
grep -n "segment_frames\|FALLBACK_FRAMES\|SPEC_MIN_FREE\|DEV_MIN_FREE\|FLOAT_DUST_EPSILON\|1e-9" voyage/config.py voyage/cli.py voyage/tui_state.py voyage/backends.py
```

## Fix candidates

1. Derive streaming/state/CUDA/worker-module sets from `BACKEND_REGISTRY`; delete hand literals + both preset wrappers after repointing `cli.py:986`, `tui_state.py:334`.
2. One TOML escaper (keep stronger C0 version, alias other); one canonical augment-floor triple + derived views; centralize geometry/latent per-backend; single free-space default + init override flag; import `FLOAT_DUST_EPSILON` in cli.
3. Import-time assertion/test that derived sets equal registry projection.
4. Gate: `gates.sh` green + `test_backend_registry`, `test_config_resolution`, `test_augment_config` green.

## Refs

- Issues 020 (escaper), 021 (CUDA pollution), 022 (caption pins precedent), 023 (streaming owner), 025 (registries)
