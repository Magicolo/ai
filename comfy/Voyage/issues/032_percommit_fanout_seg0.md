# 032 — Per-commit synchronous fan-out: 3 health RPCs + seg0 re-decode + full-frame metrics decode

- Status: open
- Severity: medium (seconds per segment + unbounded metrics RAM + O(N) seg0 decodes)
- Area: performance — commit tail (`_sample_gauges`, `_run_previous_inspect`,
  `sample_frames`)
- Rank rationale: gauges block on 3 JSONL roundtrips; inspector re-decodes all of
  seg0 on *every* segment; `sample_frames` buffers the entire raw stream.

## Technical description

- `voyage/supervisor.py:342-374` (`_sample_gauges`: 3 blocking
  `worker.call("health")` per segment, each taking the worker `_call_lock`).
- `voyage/supervisor.py:890-918` (`_run_previous_inspect`: `sample_frames(prev,5)`
  + `sample_frames(seg0,1)` **every** segment when the inspector is on).
- `voyage/vision/metrics.py:31-87` (`sample_frames`: full `ffmpeg … -f rawvideo -`
  decode piped to RAM, **all** frames, then picks 3–5; `stride = W*H*3`,
  `np.frombuffer` per frame + `.copy()`). Decodes at full segment resolution, buffers
  the entire raw stream in `proc.stdout` (unbounded RAM for long segments), copies
  every frame twice.

## Why this is an issue

A few seconds of synchronous overhead per commit is a tax on every segment of every voyage, and the seg0 re-decode makes it O(N) — the longer the run, the more decode work each new segment pays for history it already measured. Buffering the entire raw stream in RAM makes metrics memory scale with segment length instead of staying flat, risking stalls on long segments. These costs are invisible per-segment and punishing in aggregate.

## Evidence

`rg -n "_sample_gauges|sample_frames|seg0_video" Voyage/voyage/supervisor.py
Voyage/voyage/vision/metrics.py`.

Verified live 2026-09-25:

```
supervisor.py:342: def _sample_gauges ... 895: frames = sample_frames(prev_video, 5)
900: seg0_video = paths.segment_dir(...format_segment_id(0)) / "video.mp4"
901: reference = frame_histogram(sample_frames(seg0_video, 1)[0])
1247: self._sample_gauges(segment_id)
vision/metrics.py:31:def sample_frames(video_path, count=3, width=160) ...
```

## Reproduction

Inspector-on multi-segment run; observe seg0 decode wall per segment and metrics
RSS vs segment length.

## Source references

- Files/lines above.

## Resolution candidates

1. Cache the seg0 anchor histogram after segment 0.
2. Use `select='eq(n\,X)+...'` filter to decode only the 3–5 needed frames.
3. Make gauges fire-and-forget (or sample every K segments).

Payoff: removes O(N) seg0 decodes, bounds metrics RAM, cuts ~seconds/segment of
subprocess+memcpy overhead.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; _sample_gauges /
  seg0 / sample_frames refs re-verified live, current (`supervisor.py:342,895,
  900-901,1247`); pasted rg output into Evidence.
- Open: implement + measure.
