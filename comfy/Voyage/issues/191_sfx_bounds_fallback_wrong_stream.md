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

## Progress log

- 2026-09-30 (Group E2): evaluated live first. BOTH premises CONFIRMED as-read: `voyage/sfx_finalize.py:~162-171` reads `probe(segment / "video.mp4").get("streams", [{}])` → `streams[0]` with no `codec_type` filter (tree-unique — every other probe reader searches `codec_type == "video"`), and on total probe failure appends `(cursor, cursor + 0.0, "")` without advancing the cursor (verified against live file; zero-duration tiling shifts every downstream caption). `sfx_finalize.py` is hot-concurrent and explicitly out of this group's scope — logged as residual with exact lines, no code touched.

## Resolution

- Verdict: RESIDUAL — fully verified, not ownable from this group's files (the bounds planner lives entirely in `sfx_finalize.py`; no worker/audio/vision/bench/doctor seam can fix a wrong-stream read or a silent zero-tile).
- Files changed: none (this issue file only).
- Test evidence: live reads 2026-09-30 (cites above); no test added — the pins (audio-first listing → video `nb_frames`; all-unprobable → loud instead of zero-tiling; `fps > 0` validation at both functions) belong to the sfx_finalize owner with the module's probe stubs.
- DESIGN proposal (quoted text only, for the DESIGN owner — three-caption doctrine): "SFX segment bounds search the video stream (`codec_type == video`, the house idiom) and fail loud (`MediaError`, matching `_segment_timeline`) on zero-duration bounds — every later junction caption shifts otherwise, defeating the pass while reporting success."
- Residuals (for the sfx_finalize owner, precise): (1) `voyage/sfx_finalize.py:~162-164`: replace `streams[0]` with the `_probe_video_geometry` `next(... codec_type == "video" ...)` idiom; (2) zero-duration bound → `MediaError` (or at minimum a segment-naming warning — silent tiling is the current behavior); (3) validate `fps > 0` at `segment_sfx_bounds` + `_segment_timeline` entries (currently `ZeroDivisionError` vs silent-zero split).
