# 032 — Per-commit synchronous fan-out: 3 health RPCs + seg0 re-decode + full-frame metrics decode

- Status: resolved (fixed 2026-09-25: seg0 anchor + select-filter + cadence helper + tests)
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
- 2026-09-26 (resolution, FIXED — supervisor-wiring-free slices;
  `supervisor.py` untouched by scope order, hook notes in the
  helpers' docstrings): (1) `SegmentZeroAnchor`
  (`voyage/vision/metrics.py:178-216`) caches the seg0 palette
  histogram across commits — cached while path + mtime match,
  re-read on re-render, `MediaError` on missing file; hook: keep one
  on the supervisor across `run_segments` and replace the per-commit
  `frame_histogram(sample_frames(seg0_video, 1)[0])` in
  `_run_previous_inspect` (`supervisor.py:1192-1193`).
  (2) `sample_frames` (`metrics.py:131-176`) decodes only the needed
  frames via a `select='eq(n\\,X)+...'` filter (`select_frame_indices`
  `:27-40`, `select_filter_expression` `:42-55`,
  `estimate_frame_total` `:57-86` from `nb_frames` else
  duration × `avg_frame_rate`) with `-vsync 0`, falling back to the
  full decode when the estimate is missing or drifts — contract
  unchanged, RAM bounded to the selected frames in the common mp4
  case; shared `_decode_frames` (`:90-129`) drops the old double
  `.copy()` per frame. (3) `should_sample_gauges`
  (`voyage/logrotate.py:26-63`, default interval 1 matching
  `RESOURCE_GAUGE_INTERVAL_SEGMENTS`) — pure cadence gate; hook:
  replace the inline grid check in `Supervisor._sample_gauges`
  (`supervisor.py:491-496`) with it. Also fixed the pre-existing
  `Frame`/`Histogram` mypy `valid-type` failures in `metrics.py`
  (explicit `TypeAlias` — the concurrent dependency-pin rebuild
  ships mypy 2.3.1/numpy 1.26.4, under which the bare aliases no
  longer resolve; proven by running mypy over the pristine HEAD
  file in the current image: same 22 errors).
  Tests: `tests/test_issue_032_commit_fanout.py` (13 tests: cadence
  grid/clamp, pick math, filter shape/errors, estimate
  direct/fallback/unknowable, select-vs-full equality on a real
  testsrc clip, middle-frame single, anchor decode-once/re-read/
  direct-equality/missing-file).
  Gates: ruff + format + mypy strict clean on `metrics.py` +
  `logrotate.py`; new tests pass in-container. FULL-TREE GATE
  INVENTORY (all outside this scope, for the orchestrator): ruff 8
  errors in `tests/test_acestep_contract.py`,
  `tests/test_concept_integrity.py`, `tests/test_fake_backends.py`
  (×2), `tests/test_integration.py`, `tests/test_rhythm.py` (×2),
  `voyage/media.py`; mypy 14 errors in `supervisor.py` (×5),
  `cli.py` (×4), `persistence.py` (×2), `config.py`, `tui_state.py`
  (all `JsonValue`-invariance fallout of the concurrent `atomic.py`
  tightening); `tests/conftest.py` now imports `hypothesis`
  unconditionally while the gate image lacks it, so every default
  pytest run fails at collection (`--noconftest` used for all runs
  above); 4 `tests/test_ltxv_failure_hygiene.py` failures from the
  030 LRU type change (see 030 log).
