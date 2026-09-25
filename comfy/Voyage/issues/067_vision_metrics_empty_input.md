# 067 — `vision/metrics.py` empty-input crashes + NaN histogram (inconsistent guards)

- Status: resolved (fixed 2026-09-25)
- Severity: medium (typed-error contract missing on pure functions the
  supervisor calls with sampled frames)
- Area: vision metrics — `voyage/vision/metrics.py:90-97,122-130,161`
- Rank rationale: pass-2 worker/vision finding; four sibling functions guard
  empties, three don't.

## Technical description

- `:122-130` (`visual_complexity`): `return float(sum(scored)/len(scored))` with
  no `len<1` guard — unlike `motion_energy:110-113`,
  `semantic_change_rate:135-137`, `palette_distance:142-143`,
  `scene_boundary_strength:167-168`, which all return `0.0`.
- `:161` (`style_similarity`): `mid = frames[len(frames)//2]` — `IndexError` on
  `[]` when reference is not None.
- `:90-97` (`frame_histogram`): division by per-channel sum; empty frame → `0/0`
  → NaN array with only a `RuntimeWarning`.

```python
def visual_complexity(frames: list[Frame]) -> float:
    scored = []
    for frame in frames:
        ...
    return float(sum(scored) / len(scored))
```

## Why this is an issue

Four sibling functions return `0.0` on empty input while three raise
(`ZeroDivisionError`, `IndexError`) or emit `NaN` with only a warning — so the
supervisor's frame-sampling code cannot rely on any uniform contract, and a NaN
histogram propagates silently into drift and style decisions. Pure functions
this central should either all guard or all raise a typed error, not mix both.

## Evidence (live probes by pass-2 sweep)

```
visual_complexity([]) RAISED ZeroDivisionError division by zero
summarize_segment([], None) RAISED ZeroDivisionError division by zero
motion_energy([]) = 0.0
style_similarity([],ref) RAISED IndexError list index out of range
.../metrics.py:97: RuntimeWarning: invalid value encountered in divide
hist empty-frame = [nan nan ...]
```

## Reproduction

`visual_complexity([])`, `summarize_segment([], None)`,
`style_similarity([], ref)`, `frame_histogram(np.zeros((0,0,3),np.uint8))`.

## Source references

- Files/lines above.

## Resolution candidates

`if not frames: return 0.0` (or raise `MediaError`) in `visual_complexity`/
`style_similarity`/`summarize_segment`; `if frame.size==0: raise MediaError` in
`frame_histogram`. Add empty-input tests per function.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`metrics.py:90-97` histogram, `:122-130` complexity,
  `:161` style-similarity mid-frame — all match; guarding siblings at
  `:110-113`/`:135-137`/`:142-143`/`:167-168` confirmed).
- Open: implement + tests.
- 2026-09-25: FIXED in `voyage/vision/metrics.py` with the guard-everywhere
  contract: `visual_complexity([])` → 0.0 (`:130`, like the
  motion/palette/boundary siblings), `style_similarity([], ref)` → 0.0
  (`:171`, no `IndexError`; `None` ref still → 1.0), so
  `summarize_segment([], ...)` degrades gracefully instead of raising
  halfway; `frame_histogram` on a zero-size frame raises `MediaError`
  (`:94`) instead of emitting a NaN array with only a warning. Tests: 4
  new cases in `Voyage/tests/test_vision_metrics.py`
  (complexity-empty, similarity-empty-anchored/unanchored,
  summarize-empty exact dict, histogram-empty twice). Gates: `ruff check`
  clean, `ruff format --check voyage tests` clean, `mypy voyage` strict
  clean (40 files), `pytest` 472 passed.
