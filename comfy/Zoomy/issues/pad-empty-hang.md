# `_pad_frames_to_minimum` hangs forever on empty input

- Severity: high (hang — the finalize generator never yields, the Gradio
  queue slot blocks until server restart).
- Status: verified open. `zoomy/local_engine.py:1272-1283`.

## Evidence

```python
def _pad_frames_to_minimum(frames: list[Any], minimum_frames: int) -> list[Any]:
    padded = list(frames)
    while len(padded) < minimum_frames:
        padded.extend(frames)
    return padded
```

With `frames == []` and `minimum_frames >= 1`, `padded.extend([])` never
grows `padded`: the `while` spins forever. The docstring says "callers
guarantee a non-empty batch", but there is no `assert`/`raise`, and both
call sites (`_render_sound_effects`, via `_effects_stack_for` video-tensor
path at `local_engine.py:886-888`, and any future caller) can reach it with
an empty list — e.g. the single-frame finalize path yields zero
interpolated frames (see `single-frame-finalize-index-error.md`), which
would hang here instead of crashing if the order of operations ever
swapped.

## Fix

Raise at the boundary; never spin:

```python
if not frames:
    raise EngineConfigurationError("Cannot pad an empty frame batch")
```

(or `ValueError` — this is a pure helper; pick the one the module's
error taxonomy assigns to caller bugs and pin it in a test).

## Verification

- Unit test: `_pad_frames_to_minimum([], 17)` raises (currently hangs —
  run with a timeout to watch it fail first, per the TDD agreement).
- Property test: `minimum_frames >= 1`, non-empty batch → length contract
  `max(len(batch), minimum)`.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
