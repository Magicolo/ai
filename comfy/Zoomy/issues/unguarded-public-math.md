# Public duration math lacks the guards its siblings have

- Severity: medium (validation — negative/NaN inputs produce nonsense
  durations deep in the pipeline).
- Status: FIXED. All three guards mirror the established `ValueError`
  pattern (NaN-safe `not x >= ...` comparisons, same as
  `compute_frames_for_seconds`): `compute_interpolated_frame_count` and
  `compute_audio_seconds` reject counts below 1;
  `resolve_crossfade_seconds` rejects negative overlaps and non-positive
  neighbors (NaN fails the comparisons and raises instead of poisoning the
  ffmpeg filter). `compute_segment_video/music/sound_seconds` inherit the
  base guard. Tests: 3+3+3+4 rejection params plus a frames→interp→seconds
  monotonicity property. `zoomy/engine_protocol.py:215-271`,
  `zoomy/final_assembly.py:107-133`.

## Evidence

`compute_segment_windows` and the new `compute_frames_for_seconds` raise
`ValueError` on non-positive inputs — but their siblings do not:

- `compute_interpolated_frame_count(0)` → `(0-1)*4+1 = -3` (negative frame
  count flows into audio/segment math).
- `compute_audio_seconds(-100)` → `max(-3.125, 1.0) = 1.0` — silently
  "valid", masking a caller bug.
- `resolve_crossfade_seconds(nan, …)` / negative `requested_seconds` →
  `min()` propagates NaN/negative into the ffmpeg fade filter, failing far
  from the cause.

`compute_segment_video/music/sound_seconds` inherit the first two gaps.

## Fix

Mirror the established pattern: `ValueError` on `frame_count < 1`
(`compute_interpolated_frame_count`), on negative interpolated counts
(`compute_audio_seconds`), and on non-finite/negative
`resolve_crossfade_seconds` inputs. These are pure functions — pin each
guard with a direct unit test plus a Hypothesis round-trip (frames →
interp → seconds is monotonic).

## Verification

- New tests: `0`/negative/NaN inputs raise at the boundary that receives
  them.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
