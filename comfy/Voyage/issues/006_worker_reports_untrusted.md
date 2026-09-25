# 006 — Worker-reported `frames`/`recovery_path` trusted blindly (no bounds, no containment)

- Status: open
- Severity: major (DoS / deserialization oracle / state corruption)
- Area: correctness — supervisor/worker trust boundary
- Rank rationale: a buggy (or compromised) worker can exhaust disk, corrupt the
  timeline, or point resume at `/etc/passwd`.

## Technical description

```python
# voyage/supervisor.py:1145-1154
frames = config.video.segment_frames
video_block = video_result.get("video")
if isinstance(video_block, dict):
    reported = video_block.get("frames")
    if isinstance(reported, int) and reported > 0:
        frames = reported          # any positive int accepted
    tape = video_block.get("recovery_path")
    if isinstance(tape, str):
        recovery_tape = tape       # any string accepted, no exists/containment check
```

- `frames=10**9` sets `duration=frames/fps` huge → `_ensure_audio_coverage` loops
  `while cursor < end` slicing thousands of pieces (disk exhaustion / hours of
  ffmpeg), `timeline_frames` corrupts, `beats_for_segment` garbage (and see 039:
  `inf` input already produces a 5.6e162 beat count — huge `frames` is the
  realistic trigger).
- `recovery_path="/etc/passwd"` (or nonexistent) is stored into `metrics.json` and
  later passed to `resume`/`rebuild` — for torch-based workers `recovery.pt` is
  unpickled (see 005), so a foreign path is at minimum 3 wasted restart-budget
  retries, at worst a deserialization oracle. `validate` later checks
  `Path(tape).exists()` but commit does not.

## Why this is an issue

- The supervisor/worker wire is a trust boundary (separate processes, separate
  images, upstream model code). Blind trust here voids the restart-budget and
  tape-containment assumptions the crash matrix relies on.
- Failure is late and expensive (disk fill / retry storms) instead of immediate.

## Evidence

Source quotes above; `validate`'s later `exists()` check (`cli.py:549-551`)
proves commit-time validation is missing, not impossible.

Live frame-accounting site (re-verified 2026-09-25):

```
$ sed -n '1145,1156p' voyage/supervisor.py
        frames = config.video.segment_frames
        video_block = video_result.get("video")
        recovery_tape: str | None = None
        if isinstance(video_block, dict):
            reported = video_block.get("frames")
            if isinstance(reported, int) and reported > 0:
                frames = reported          # any positive int accepted
            tape = video_block.get("recovery_path")
            if isinstance(tape, str):
                recovery_tape = tape       # any string accepted
```

## Reproduction

1. Stub video worker returning `{"video": {"frames": 10**9}}` → commit attempts
   ~1e9/24 s of audio coverage.
2. Stub returning `{"video": {"frames": 48, "recovery_path": "/nonexistent/x.pt"}}`
   → `rebuild` retries then FAILED, where a containment check would reject at once.

## Source references

- `voyage/supervisor.py:1145-1154`; `voyage/cli.py:549-551` (validate-side check);
  `voyage/audio/beat.py:22-58` (downstream consumer of huge durations).

## Resolution candidates

1. Clamp `reported` to a sane range (e.g. `1..10*config.video.segment_frames` or a
   backend-specific max), else `MediaError` immediately.
2. Validate `recovery_path` is under `run_dir` (or `models_dir`) and exists before
   accepting, else `MediaError`. Persist run-relative paths (see 016).
3. Tests: oversized-frames stub and foreign-tape stub both fail fast at commit.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Open: implement + tests.
- 2026-09-25 (repair pass): refs verified current (`supervisor.py:1149-1154`;
  `cli.py:551`); Evidence enriched with live `sed` output;
  `## Why this is an issue` already present, no change.
