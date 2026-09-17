# Silent video and stems leak on music/SFX/mux failure

- Severity: medium (disk leak on exactly the failure paths that need
  retried renders).
- Status: verified open. `zoomy/local_engine.py:381-404`.

## Evidence

`silent_video_path.unlink(missing_ok=True)` runs only on the success path
(line 404). If `_render_music`, `_render_sound_effects`, or either
`_mux_audio_twin` raises, the function exits via exception with the silent
video (hundreds of MB on long windows) plus any landed stems left in
`output_directory` / the assembly workdir. The retry wrapper
(`run_stage_with_retries`) then re-renders the window, orphaning the
previous attempt's files permanently — `remove_segment_files` only runs
after a fully successful `finalize_sequence` (`local_engine.py:338`).

## Fix

Delete the window's partial artifacts in a `finally` (or on the failure
path before re-raise): silent video + both stems + both twins for this
`window.index`. Keep the successful outputs untouched. Test with stubbed
`_render_music` raising: assert no `*_seg000_*` files remain and the
surfaced error is unchanged.

## Verification

- New test: failing music stage → `AssemblyError`/`EngineExecutionError`
  propagates AND zero segment files for that window remain.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
