# Single-frame finalize crashes with a cryptic `IndexError`

- Severity: high (correctness — the smallest legal sequence cannot finalize).
- Status: FIXED. `<2`-frame passthrough landed in the concurrent
  `a3aa393` with `test_interpolation_passes_single_frame_through`; this
  change adds the 1-frame window e2e with stubbed audio
  (`test_single_frame_window_finalizes_with_stubbed_audio`, twins land on
  disk). `zoomy/local_engine.py:764-765`.

## Evidence

`_interpolate_frames` expands pairs:

```python
for first, second in pairwise(arrays):
    ...
```

With one source frame, `pairwise` yields nothing, `interpolated` stays
`[]`, and `_write_silent_video` immediately does
`np.asarray(frames[0])` → `IndexError`. The generic handler in
`render_window` (`local_engine.py:412-414`) wraps it as
`EngineExecutionError("segment-0", "Segment 0 failed: list index out of
range")` — nothing tells the user the sequence was too short, and the
`finalize_video` docstring (`rendering.py:142-166`) promises any non-empty
sequence finalizes. Note the sibling trap: an empty interpolated list
passed to `_pad_frames_to_minimum` would *hang* instead (see
`pad-empty-hang.md`).

## Fix

Passthrough for `< 2` frames at the top of `_interpolate_frames`
(`return list(source_frames)`), matching the former Comfy FILM behavior
the design doc describes. Add a regression test finalizing a 1-frame
sequence through the single-pass path with stubbed music/SFX (assert a
valid mp4 + audio twin, not just "no crash").

## Verification

- New test: 1-frame `finalize_sequence` succeeds end to end (fakes for the
  audio stages).
- Existing duration math already covers it:
  `compute_interpolated_frame_count(1) == 1`.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
