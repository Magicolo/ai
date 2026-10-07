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
