# 248 — Duration math has two frame-count sources + unguarded division; CUDA sets computed at import; dead `else` label in `_require_cuda_stack`

Severity: MEDIUM (track B-12).

## Technical description

`_frames_per_segment` multiplies cli-local novel constants (`_LTXV_NOVEL_BLOCK_FRAMES=96`,
`_CAUSVID_NOVEL_PER_ROLLOUT=72`, `_LTX25_NOVEL_BLOCK_FRAMES=232`,
`_LTX_NOVEL_BLOCK_FRAMES=96`) while `BACKEND_REGISTRY.segment_frames` carries
`96/72/232/96` for the same backends — values agree today by hand, with no pin.
`segments_for_duration` divides by `frames_per_segment` with no zero guard
(`ZeroDivisionError`, not `ValueError`). `_CUDA_VIDEO/AUDIO/SFX_BACKENDS` are
import-time frozensets over a mutable `BACKEND_REGISTRY` dict. `_require_cuda_stack`'s
`else config.video.backend` fallback is unreachable (`needs_cuda` implies non-empty
`offenders`).

## Rationale

Duplicated planning constants + import-time derivation + dead fallback = three small
drift/crash vectors in the preflight every `generate` runs.

## Live evidence

```
_LTXV_NOVEL_BLOCK_FRAMES = 96 vs registry ltxv segment_frames=96 (and 72/232/96 pairs)  # cli_planning.py:80-99 vs config.py:137,152,170,209,224
def segments_for_duration(...): return max(1, math.ceil(duration_seconds*fps/frames_per_segment - EPS))  # cli_planning.py:115-117, no zero check
_CUDA_VIDEO_BACKENDS = frozenset(... BACKEND_REGISTRY.items() ...)  # cli_planning.py:120-137
label = " + ".join(offenders) if offenders else config.video.backend  # :206, else dead
```

Repro: `segments_for_duration(10.0, 24, 0)` → `ZeroDivisionError`; mutate
`BACKEND_REGISTRY["fake"]` at runtime → `_CUDA_*` stale; valid CUDA config never hits the
`else` branch (cover with a test to prove dead).

## Source refs

`voyage/cli_planning.py:74-145,148-208`; `voyage/config.py:117-234`.

## Online sources

- None (in-tree registry table is the single-source-of-truth candidate).

## Fix candidates

- Derive novel counts from `BACKEND_REGISTRY` (single source) or pin equality in
  `test_backend_registry.py`; `ValueError` on `frames_per_segment<=0`/`fps<=0`; build CUDA
  sets via function (not import-time constant) or freeze the registry
  (`MappingProxyType`); delete the dead `else`.

## Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.

## Evaluation (2026-10-07)
- Re-read `voyage/cli_planning.py:74-208` + `voyage/config.py:117-234`
  live: all four pairs agree today (96/96, 72/72, 232/232, 96/96) by hand
  with no pin — NOT stale. `segments_for_duration(10.0, 24, 0)` raises raw
  `ZeroDivisionError` in-container (confirmed) — NOT stale. CUDA sets are
  import-time frozensets over the mutable registry dict — live. The
  `else config.video.backend` fallback is unreachable (`needs_cuda` true
  implies non-empty offenders under the same sets) — live dead code.
- Existing pins checked first: `test_backend_registry.py` pins geometry/
  audio pairing + streaming derivation but NOT novel counts;
  `test_generate.py` pins `_frames_per_segment` outputs (96/232/72) but not
  their source; `test_single_source.py` pins epsilon/streaming/presets but
  not novel counts or CUDA-set freshness. Derivation (not pin-only) is
  safe: no test imports the four cli-local constants directly (grep proves
  only `cli_planning.py` references them).

## Progress log (2026-10-07)
- `_frames_per_segment` now reads `BACKEND_REGISTRY[backend].segment_frames
  * blocks_per_segment` at call time for the four streaming backends
  (single source; the four cli-local constants deleted, provenance comments
  folded into the docstring); `fake` still uses stored `segment_frames`.
- `segments_for_duration` raises `ValueError` on `frames_per_segment <= 0`
  (bool-safe) and `fps <= 0` — the `ZeroDivisionError` vector is gone.
- CUDA sets: new `_cuda_video/audio/sfx_backends()` functions read the
  registry fresh; `_cuda_offenders` + `_require_cuda_stack` consume them.
  The `_CUDA_*` import-time frozensets stay as legacy snapshots for
  `test_surface_rank2` compat (that file is outside this scope and pins
  them directly) with docstrings pointing at the functions. Full removal
  of the constants is a follow-up owned with that test.
- Deleted the dead `else config.video.backend` fallback (comment records
  why `offenders` is non-empty whenever reached).
- `tests/test_backend_registry.py`: new `test_planning_novel_counts_derive_from_registry_248`
  (per-backend `_frames_per_segment == registry row`, plus a 2-block pin)
  and `test_segments_for_duration_rejects_non_positive_248`.

## Resolution (2026-10-07)
- RESOLVED. Files: `voyage/cli_planning.py`,
  `tests/test_backend_registry.py`. Default planning outputs verified
  unchanged in-container (fake 48 / ltxv 96 / causvid 72 / ltx25 232 /
  ltx23 96 at blocks 1; sample `segments_for_duration` values identical
  before/after). Scoped + neighbor pytest green; ruff + format + mypy
  strict clean on touched modules.
- Left open: full removal of the legacy `_CUDA_*` snapshots (needs a
  `test_surface_rank2.py` update, outside this scope — that file still
  pins the constants directly and passes).
