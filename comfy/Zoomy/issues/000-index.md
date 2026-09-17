# Zoomy issue index

Per-bug files. Every issue below was reverified against the working tree
**after** the concurrent uncommitted work (frame-size validation,
`run_stage_with_retries`, segment-stem rename, `generate_video`,
`compute_frames_for_seconds`, `quality-gates.sh`/`run-tests.sh`) — only
still-open findings are filed.

## Verified fixed by concurrent work (do NOT re-file)

- Segment-file prefix mismatch (`Zoomy_` vs bare): fixed —
  `local_engine.py:372` now emits `f"{sequence_key}_seg…"`, matching
  `frame_repository.py:114,126,156`.
- `FrameRenderRequest.frame_width/frame_height` ignored: fixed —
  `local_engine.py:217,232-245` validates alignment, `_zoomed_frame` and
  the Z call take the request size, interface coerces via
  `_coerce_frame_size` (`interface.py:788`).
- VRAM unknown shown as `0 GiB`: fixed for VRAM —
  `engine_statistics` (`local_engine.py:186-204`) reports `None` → `n/a`.
  System RAM still reports `(0, 0)` (see `system-memory-zero-not-na.md`).
- `compute_frames_for_seconds` unguarded: fixed with tests
  (`engine_protocol.py:217-232`, `test_engine_protocol.py`).
- `family_catalog.py` docstring `Zoomy_` prefix: fixed.
- `quality-gates.sh` / `run-tests.sh` exist (untracked): see
  `scripts-hardening.md` for remaining gaps.

## Open issues

### High

- `pad-empty-hang.md` — `_pad_frames_to_minimum` infinite loop on empty input.
- `single-frame-finalize-index-error.md` — 1-frame finalize crashes in
  `_write_silent_video` via `pairwise` yielding nothing.

### Medium

- `popen-stdin-write-leak.md` — ffmpeg child/pipe leaked on stdin failure.
- `silent-video-cleanup-no-finally.md` — silent video + stems leak on
  music/SFX/mux failure.
- `pil-handles-unclosed.md` — source frames opened without closing.
- `silent-model-fallbacks.md` — transformer/autoencoder failures silently
  fall back to official weights.
- `settings-whitespace-text.md` — `_read_text` treats `"   "` as configured.
- `find-family-uncaught-handlers.md` — bad dropdown key escapes handlers as
  a traceback (with `show_error=True`).
- `safe-statistics-narrow-except.md` — non-`ZoomyError` from the engine kills
  the 10 s stats timer.
- `clear-frames-traversal-ignore-errors.md` — unsanitized `sequence_key` +
  silent delete failures.
- `unguarded-public-math.md` — `compute_interpolated_frame_count(0)`,
  negative audio/segmentation inputs, `resolve_crossfade_seconds` NaN/negative.
- `dockerfile-reproducibility.md` — unpinned ACE-Step, tautological smoke
  assert, no digest, unpinned apt.
- `requirements-gpu-mixed-pins.md` — bare/`>=` pins alongside exact pins.
- `pyproject-missing-enforcement.md` — no `addopts`, no coverage gate, no
  `warn_unused_ignores`.
- `conftest-hypothesis-docstring-false.md` — docstring promise contradicted
  by `.hypothesis/` on disk.
- `scripts-hardening.md` — unquoted `$*`, missing cache/Hypothesis env in
  `run-tests.sh`.

### Low

- `gitignore-dockerignore-hygiene.md` — caches and outputs missing from
  ignore files.
- `magic-sync-constants.md` — literal `25`/`44100` duplicate
  `OUTPUT_SAMPLE_RATE`.
- `system-memory-zero-not-na.md` — `/proc` failure reports `0.0 GiB`, not
  `n/a`.
- `vendor-compat-no-lock.md` — `_APPLIED_PATCHES` mutated without a lock.
- `loop-stop-global-event.md` — module-level stop flag shared across sessions.
- `interrupt-clear-race.md` — `self._interrupt.clear()` can wipe a
  concurrently set interrupt.
- `coercion-silent-defaults.md` — garbage payloads coerce to `0.0`/`set()`.
- `output-residue-cleanup.md` — e2e mp4s accumulating under `Zoomy/output/`.
