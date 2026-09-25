# 031 — Finalize does 2 full video encodes + re-slices all audio from takes

- Status: open
- Severity: medium (finalize wall; O(segments × pieces) ffmpeg spawns)
- Area: performance — `voyage/media.py:303-549`
- Rank rationale: the first pass is pure waste on native-geometry runs (now the
  default); audio re-slicing repeats identical bytes across overlapping windows.

## Technical description

Per segment: video-only `libx264 veryfast` pass → concat demuxer → **second**
`libx264 veryfast + scale/pad/fps + AAC` pass (`media.py:469-492` parts loop +
`:511-544` final encode). When generation geometry already equals finalize
geometry (now the default — `cli.py:640-648` keeps generation resolution), the
first pass is pure waste. Audio: `build_final_audio` re-`slice_take`s every window
from take WAVs (`-ss/-t` per piece, each a separate ffmpeg invocation,
`:354-388`) then chains `acrossfade`s — O(segments × pieces) ffmpeg spawns, each
reopening multi-GB takes. Single-segment fast path copies; multi-segment never
copies even when `overlap_fraction=0`.

## Why this is an issue

Finalize runs once per voyage but its wall time scales with total footage, so a wasted full video encode plus O(segments × pieces) ffmpeg spawns — each reopening multi-GB take files — lands hardest on exactly the long runs users care about. Now that native geometry is the default, the first pass is pure waste on the common path: ~50% of finalize time burned re-encoding bytes that already match. Every future finalize optimization has to work around this redundant pass until it is removed.

## Evidence

Read `media.py:303-403,466-549`; count ffmpeg spawns in a 10-seg finalize:
`N(video parts) + N(windows × pieces) + 1(concat) + 1(final)`.

Verified live 2026-09-25:

```
406:def finalize_run(
407-    run_dir: Path, 408: output_path: Path, 409: width: int = 768,
410-    height: int = 432, 411: fps: int = 24, 412: skip_bad: bool = False,
413-    min_free_space_gib: float = 0.0, 414: sample_rate: int = 48000,
415-    channels: int = 2, 416: overlap_fraction: float = 0.10,
417-    overlap_cap_seconds: float = 0.5, 418:) -> Path:
```

## Reproduction

Finalize a 10-segment native-geometry run; count ffmpeg invocations and compare
against a single-encode baseline.

## Source references

- `voyage/media.py:303-403,466-549`; `voyage/cli.py:640-648`.

## Resolution candidates

1. If all parts already match `WxH/fps/pix_fmt`, `ffmpeg -f concat -c copy` (zero
   re-encode), single scale/encode only if needed. Saves ~50% of finalize wall on
   native runs.
2. Cache take slices by `(take_id, start, dur)` — windows overlap by `half`, so
   adjacent segments re-slice identical bytes today.
3. Offer NVENC (`h264_nvenc`) when available; `veryfast` x264 on CPU is the
   finalize bottleneck for long runs (unmeasured — see bench gaps in 041/049
   context: no encode A/B exists).

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; `finalize_run`
  signature re-verified live (`media.py:406-418`, current); pasted output into
  Evidence.
- Open: implement concat-copy fast path + slice cache + measurement.
