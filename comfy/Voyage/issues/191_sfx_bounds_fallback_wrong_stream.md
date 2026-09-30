# 191 — `segment_sfx_bounds` fallback chain reads the wrong stream and tiles silent zero-length bounds

- Severity: LOW-MEDIUM
- Area: SFX finalize — caption-timeline correctness
- Files (as-read 2026-09-30):
  - `voyage/sfx_finalize.py:154-177` (`segment_sfx_bounds`: probe → `streams[0]` fallback → caption read)
  - `voyage/media.py:363-384` (`_segment_timeline`: the strict counterpart — raises on frames <= 0)
  - `voyage/media.py:754-771` (`_probe_video_geometry`: the video-stream-search precedent)

## Description

Two independent defects in the SFX bounds fallback, both silent, both shifting
every downstream caption:

1. **First-stream instead of video-stream (`sfx_finalize.py:162-164`).** When the
   container-duration probe fails, the fallback reads `streams[0]` — whatever
   ffprobe lists first, which for muxed segments is routinely the *audio* stream
   — and takes its `nb_frames`:

   ```python
   streams = probe(segment / "video.mp4").get("streams", [{}])
   first = streams[0] if streams else {}
   frames = int(first.get("nb_frames", 0) if isinstance(first, dict) else 0)
   ```

   Every other probe reader in the tree searches `codec_type == "video"`
   (`media.py:106,132,762,843`). An audio-first listing yields the audio
   stream's `nb_frames` (or 0 when absent) divided by the *video* fps — a
   duration that is wrong but plausible, so the bound is accepted.

2. **Zero-duration bounds tile silently (`:176-177`).** When the fallback also
   yields 0, the segment gets a `(cursor, cursor, "")` bound and the cursor does
   not advance. All later segments' bounds shift early relative to the real
   timeline, so `plan_sfx_windows` assigns junction captions to the wrong windows
   for the rest of the run — the SFX renders hear the wrong transition. The
   strict counterpart `_segment_timeline` raises `MediaError` on `frames <= 0`
   (`media.py:379-380`); the SFX path degrades to mis-captioned audio with no
   error, no warning, and a bed that still passes the timeline-exactness check
   (`:410-412`, which compares against the *final* duration, not the bounds).

## Rationale

Caption placement is the SFX pass's reason to exist (junction windows "hear both
sides" — `plan_sfx_windows:83-92`). A fallback that silently misplaces every
caption downstream of one unprobable segment defeats the pass while reporting
success. The fix is two lines plus a warning, and the precedent
(`_probe_video_geometry`, `_probe_video_fps`) already shows the house pattern.

## Live evidence (verified live 2026-09-30, host reads + probe)

- `sed -n '154,177p' voyage/sfx_finalize.py` — `streams[0]` with no `codec_type`
  filter; `bounds.append((cursor, cursor + duration, …))` unconditional on
  `duration == 0.0`.
- `rg -n 'codec_type.*video' voyage/media.py` → 4 video-stream searches; `rg -n
  'streams\[0\]' voyage/` → this site only. The inconsistency is tree-unique.
- `sed -n '375,380p' voyage/media.py` — `_segment_timeline` raises on
  unreadable/non-positive frames: same corrupt input, opposite loudness.
- Live contrast probe (host python3, stdlib only): `_segment_timeline([dir], 0)`
  raises raw `ZeroDivisionError` while the SFX bounds path guards only
  `frames > 0` — neither path validates the fps it divides by (secondary
  edge, same function pair).

## Repro

1. Craft a segment whose `video.mp4` probes with an audio stream first (or stub
   `probe` to return `{"streams": [{"codec_type": "audio", "nb_frames": "300"},
   {"codec_type": "video", …}], "format": {"duration": 0}}`).
2. `segment_sfx_bounds(run_dir, [seg], 24)` → duration from the audio stream's
   300 frames at video fps instead of the video stream's count.
3. Stub `probe` to raise `MediaError` throughout → bound `(0.0, 0.0, "")`, cursor
   stuck; a 3-segment run gets bounds `(0,0),(0,0),(0,0)` and every window
   carries the joined captions of all segments.

## Fix candidates

1. Search the video stream in the fallback (copy `_probe_video_geometry`'s
   `next(… codec_type == "video" …)` idiom); treat "no video stream" as
   `duration = 0.0` only after the search fails.
2. Fail loud (or warn loud) on a zero-duration bound: `MediaError` matches the
   `_segment_timeline` contract; at minimum a `print`/metric warning naming the
   segment, since every later caption shifts.
3. Validate `fps > 0` at both functions' entries (`ValueError`/`MediaError`,
   not the current `ZeroDivisionError` vs silent-zero split).
4. Tests: audio-first stream listing uses video `nb_frames`; all-unprobable run
   raises instead of tiling zeros.

## Refs

- In-tree: `voyage/sfx_finalize.py:135-177,293` (plan consumes bounds);
  `voyage/media.py:363-384,754-771`; `DESIGN` three-caption doctrine (junction
  windows).
- Not-a-duplicate: 003 is A/V alignment budgets; 138 is finalize skip legs
  (segment-level, not caption-bound legs); 153 is stem cache/unlink/truncate
  (render reuse, not bounds planning); 126 is vision-metric edge inputs (different
  module, different crash class).
