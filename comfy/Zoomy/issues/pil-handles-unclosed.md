# Source frames opened without closing in `_finalize_window`

- Severity: medium (FD exhaustion on long segmented finalizes).
- Status: FIXED. New `_converted_frame` helper opens each PNG as a
  context manager and returns the converted copy; all three call sites
  (`render_window` frames, `_load_source_image` cold + warm) use it.
  Empirical note: probing showed current Pillow releases the handle during
  `load()` (FD count flat at 4 holding 48 converted copies, no
  `ResourceWarning` ever fires), so this is defense-in-depth against
  decoder internals, not a live leak — the tests pin *our* discipline via
  a close-recording `open` fake (`test_window_render_closes_source_files`,
  `test_load_source_image_closes_seed_files`), not Pillow behavior.

## Evidence

```python
source_frames = [PillowImage.open(path).convert("RGB") for path in frame_paths]
```

`PillowImage.open` is lazy — the file handle stays open until the image
is closed or GC'd. The converted copy is what the pipeline uses; the
originals are never closed. A 48-frame window holds 48 FDs through
interpolation + music + SFX + mux; a segmented finalize of hundreds of
frames repeats this per window, and CPython refcounting is the only thing
reclaiming them (no explicit `close()`, no context manager).

## Fix

```python
with contextlib.ExitStack() as stack:
    source_frames = [stack.enter_context(PillowImage.open(p)).convert("RGB") for p in frame_paths]
    ...
```

(`convert` copies pixel data, so closing the originals right after the
comprehension is equally correct and simpler.) Assert in a test that no
file handles remain open after `render_window` (e.g. `/proc/self/fd`
count or `warnings.simplefilter("error", ResourceWarning)` around the
call).

## Verification

- New test: FD count stable across a stubbed window render.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
