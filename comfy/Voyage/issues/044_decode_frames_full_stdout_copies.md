# 044 — `_decode_frames` captures the whole rawvideo stdout + duplicates every frame

**Severity:** HIGH

**File:line:** `voyage/vision/metrics.py:90-128` (`_decode_frames`), `131-175` (`sample_frames`), esp. `94-113`, `118`, `123-128`, `168`, `170-175`

**Description:**
`subprocess.run(..., capture_output=True)` accumulates the entire decoded `rawvideo` byte stream in `proc.stdout`, then `np.frombuffer` slices per-frame views, then `sample_frames` returns `[frame.copy() for frame in selected]` — so at peak: 1× full byte string + N views + N full copies. For 100f × 768×512×3 ≈ 112.5 MiB raw, held ~2× plus interpreter overhead; a longlive 1280×704 93f segment ≈ 250 MiB. The `select`-filter fast path still captures everything the filter emits, and the estimate-drift fallback (line 170) full-decodes the whole clip just to pick 3 frames. No streaming (`Popen.stdout.read(chunk)`), no `-frames:v` cap, no chunked decode.

**Rationale:**
Inspector runs per segment with no per-stage timing or RSS gauge around `sample_frames`; host-RSS growth is invisible because `bench.summarize_gauges` drops decode peaks (see 051). Best practice is chunked/streamed decode for frame sampling, never `capture_output` on video bytes.

**Live evidence (current tree):**
```python
voyage/vision/metrics.py:94: proc = subprocess.run([... "-f", "rawvideo", "-"], capture_output=True, ...)
voyage/vision/metrics.py:118: raw = proc.stdout if isinstance(proc.stdout, bytes) else b""
voyage/vision/metrics.py:119: stride = width * height * 3
voyage/vision/metrics.py:123-128: [np.frombuffer(raw[i*stride:(i+1)*stride]...).reshape(...) for i in ...]
voyage/vision/metrics.py:168: return [frame.copy() for frame in selected]
voyage/vision/metrics.py:170: frames = _decode_frames(video_path, width, height, f"scale=...")
voyage/vision/metrics.py:91: docstring "return decoded RGB frames (no copy)" — false: callers .copy() every frame
```
Size math (arithmetic, host has no numpy so not executed): 100 × 768×512×3 = 112.5 MiB in `stdout` + views + copies ≈ 225 MiB transient (host `python3 -c` probe confirms 112.5). `ffmpeg 8.0.1` present.

**Repro:**
```bash
ffmpeg -hide_banner -nostdin -y -f lavfi -i testsrc=size=768x512:rate=24:duration=4 \
  -c:v libx264 -pix_fmt yuv420p /tmp/probe.mp4
python3 -c "print((768*512*3*96)/1024**2)"  # → 108.0 MiB raw held in stdout for one 96f segment
```

**Fix candidates:**
- Stream with `Popen` + incremental `read(stride)` and yield frames; stop after the last wanted index.
- Cap the select path with `-frames:v len(picks)` so a drifted estimate cannot emit the whole stream.
- Reuse the `augment.ffmpeg_decode_chunk`-style PNG-chunk decode or keep the `SegmentZeroAnchor` cache (lines 178-217) and add an RSS gauge around `sample_frames`.

**Refs:** `voyage/vision/metrics.py:178-217` anchor (proves per-commit re-decode was already a cost center); ffmpeg filter docs (avoid full-stream materialization).

**Overlaps with:** 043/045/048 (all-at-once RAM cluster — same streaming/chunking fix pattern; not duplicates).

## Progress log

- 2026-09-30: re-verified every premise against live code — all hold as-read: `metrics.py:94` `subprocess.run(capture_output=True)`, `:118` full-stdout hold, `:123-128` `frombuffer` views, `:168`/`:173` caller `.copy()` fan-out, `:170` uncapped full-decode fallback, and the `:91` "no copy" docstring is indeed false (callers copy every frame).
- 2026-09-30 (TDD red): the 044 tests in `Voyage/tests/test_media_memory.py` failed first in-container (missing `FALLBACK_MAX_FRAMES` + `subprocess.run` still capturing rawvideo).
- 2026-09-30 (implement): rewrote `_decode_frames` around `Popen` + incremental exact-`stride` reads (`_read_frame_bytes`: clean-EOF → stop, short tail → loud `MediaError`); each frame is one owned array at decode (`.copy()` once, `bytes` chunk released) — peak is retained frames, never the stream. New keyword-only `frame_limit` appends `-frames:v` server-side and kills the process at the cap (kills are not errors; only EOF-before-limit checks the exit status). `sample_frames` passes `frame_limit=len(picks)` on the select path (a drifted estimate can emit at most `count` frames), drops the entire `.copy()` fan-out, and the fallback decodes at most `FALLBACK_MAX_FRAMES = 2048` (10x the largest known segment — real segments are < 200 frames) serving evenly spaced picks from the capped prefix.
- 2026-09-30 (TDD green): identity vs the inline old algorithm (`array_equal` on all 3 picks), two mock-gates proving no `subprocess.run` rawvideo capture (ffprobe still allowed through), drifted-estimate (100 000) still serves 3 frames, `None`-estimate fallback is frame-identical to the select path, RSS smoke (< 120 MiB growth), cap-constant pin. One test-side fix (mock-gate initially forbade the legitimate ffprobe `subprocess.run` — narrowed to rawvideo argv).
- 2026-09-30 (gates): ruff + format-check + mypy strict clean; `test_vision_metrics` + `test_issue_032_commit_fanout` (incl. the select-path tests) all pass unmodified — public signatures backward-compatible.

## Resolution

- Verdict: fixed. Peak decode memory is now the retained frames (~3 scaled frames on the select path, ≤ cap only on the rare drift path) instead of ~2-3x the whole rawvideo stream.
- Files changed: `Voyage/voyage/vision/metrics.py` (+`FALLBACK_MAX_FRAMES`, +`_read_frame_bytes`, rewrote `_decode_frames`, rewired `sample_frames`), `Voyage/tests/test_media_memory.py` (new: 7 tests listed above).
- Test evidence: `test_sample_frames_matches_reference_decode` (exact-pixel identity vs pre-044 algorithm on synthetic testsrc), `test_sample_frames_drifted_estimate_still_serves_count`, `test_sample_frames_missing_estimate_serves_identical_frames`, `test_sample_frames_bounded_rss`, `test_sample_frames_never_uses_capture_run` (+ single-frame variant).
- DESIGN.md as-built proposal (not applied): in §§43-44/100, after the sampling description, add "frame sampling streams the rawvideo decode (`Popen`, one owned array per frame, `-frames:v` early stop); the select path caps emitted frames at the pick count and the estimate-drift fallback decodes at most 2048 frames (capped prefix picks) — sampling never holds the whole stream."
- Residuals: on a > 2048-frame clip with a missing/drifted estimate, fallback picks come from the capped prefix, not the whole clip (no real segment is near this — segments are < 200 frames; deliberate bound, not a bug). No per-stage RSS gauge around `sample_frames` (that is issue 051's scope, overlapped not duplicated).
