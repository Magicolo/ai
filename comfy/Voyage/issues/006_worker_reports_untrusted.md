# 006 — Worker-reported `frames`/`recovery_path` trusted blindly (no bounds, no containment)

- Status: resolved in live tree (frame ceiling + tape containment landed)
- Severity: HIGH (DoS / deserialization oracle / state corruption; resolved, record only)
- Group: security/trust-boundary — Rank: 1/5 (critical pattern, fixed)
- Area: correctness — supervisor/worker trust boundary
- Rank rationale: a buggy (or compromised) worker could exhaust disk, corrupt the
  timeline, or point resume at `/etc/passwd`.

## Technical description

Pre-fix (`voyage/supervisor.py:1145-1154` at pass 1):

```python
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
- `recovery_path="/etc/passwd"` (or nonexistent) was stored into `metrics.json`
  and later passed to `resume`/`rebuild` — for torch-based workers the tape is
  unpickled (see 005), so a foreign path is at minimum 3 wasted restart-budget
  retries, at worst a deserialization oracle. `validate` later checked
  `Path(tape).exists()` but commit did not.

Live state (re-verified 2026-09-30): `REPORTED_FRAMES_SLACK = 10`
(`voyage/supervisor.py:132`) with a `1..10*segment_frames` ceiling
(`voyage/supervisor.py:1636-1648`, `MediaError` on violation), and
`_checked_tape_path` (`voyage/supervisor.py:403-415`, run-dir containment + exists)
gating every tape.

## Why this is an issue

- The supervisor/worker wire is a trust boundary (separate processes, separate
  images, upstream model code). Blind trust here voids the restart-budget and
  tape-containment assumptions the crash matrix relies on.
- Failure is late and expensive (disk fill / retry storms) instead of immediate.

## Evidence

Live verification 2026-09-30 (in-container):

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "..."
slack: REPORTED_FRAMES_SLACK = 10
checked-tape: True
frames-ceiling: True
```

```
$ rg -n "REPORTED_FRAMES_SLACK|_checked_tape_path|implausible" voyage/supervisor.py
132:REPORTED_FRAMES_SLACK = 10
396:    def _checked_tape_path(self, tape: str, segment_id: str) -> str:
1635:                         f"segment {segment_id}: worker reported implausible "
```

Pass-1 live site: `sed -n '1145,1156p' voyage/supervisor.py` showed the
any-positive-int / any-string acceptance quoted above.

## Reproduction

1. Stub video worker returning `{"video": {"frames": 10**9}}` → pre-fix commit
   attempted ~1e9/24 s of audio coverage; now immediate `MediaError`.
2. Stub returning `{"video": {"frames": 48, "recovery_path": "/nonexistent/x.pt"}}`
   → pre-fix `rebuild` retries then FAILED; now containment check rejects at once.

## Source references

- `voyage/supervisor.py:128-162` (ceiling contract), `:396-414` (tape check),
  `:1623-1640` (commit gating); `voyage/cli.py:808-810` (validate-side resolve);
  `voyage/audio/beat.py:61-75` (finite/take guards downstream).

## Resolution candidates

1. (Landed) Clamp `reported` to `1..10*config.video.segment_frames`
   (backend-specific max is a possible tightening), else `MediaError` immediately.
2. (Landed) Validate `recovery_path` is under `run_dir` and exists before
   accepting, else `MediaError`. Persist run-relative paths (see 016).
3. Tests: oversized-frames stub and foreign-tape stub both fail fast at commit.

## Online references

- Uncontrolled resource consumption (CWE-400 — the `frames=10**9` disk/time
  exhaustion class): https://cwe.mitre.org/data/definitions/400.html
- Allocation without limits (CWE-770):
  https://cwe.mitre.org/data/definitions/770.html
- Deserialization of untrusted data, oracle variant (CWE-502):
  https://cwe.mitre.org/data/definitions/502.html

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Resolution batch 3: frame ceiling + `_checked_tape_path` landed.
- 2026-09-30: re-verified live (slack, ceiling, containment all present);
  reconstructed from archived pass-1 text (commit `b5d7dda`). Status → resolved.

## Progress log (2026-09-30, supervisor-track resolution)

- Premise re-verified against current live code (in-container):
  `Supervisor._render_video` re-gates the raw worker report with the
  `REPORTED_FRAMES_SLACK = 10` ceiling (`1..10*segment_frames`, bool and
  non-int rejected, `MediaError` on violation) and re-resolves every
  non-empty string tape through `_checked_tape_path` (run-dir containment
  + `resolve()` + `is_file()`); `Voyage/voyage/backends.py`
  `generate_segment` normalizes leniently but the supervisor re-gates
  before trusting `returned_frames` — main path already resolved. Gap
  found: `Supervisor._adopt_unaccounted_segment` checked `frames` only
  for int/`> 0` with no ceiling, so a DONE orphan carrying
  `frames=10**9` in `metrics.json` adopted and corrupted `timeline_frames`.
- TDD: new `Voyage/tests/test_supervisor_av_align.py` —
  `test_adopt_rejects_absurd_frames` (crafted orphan, `frames=10**9`,
  valid checksums) failed first with `DID NOT RAISE MediaError`, passes
  after the fix. Characterization pins for the live clamps:
  `test_worker_reported_negative_frames_rejected`,
  `test_worker_reported_bool_frames_rejected`,
  `test_foreign_tape_directory_rejected` (all pass before and after).
- Fix: adoption path now enforces the same
  `1..REPORTED_FRAMES_SLACK*segment_frames` ceiling with the identical
  `implausible` `MediaError` before advancing state. Live-report and
  tape gates untouched (already correct).
- Evidence: new suite 8 passed; related suites as in 003 (38 + 40
  passed). `ruff check` + `ruff format --check` + `mypy strict` clean on
  touched files.

## Resolution

- Status: resolved (main path was already resolved; adoption-path ceiling
  gap closed this pass). No `DESIGN.md` edit made here — as-built proposal
  is in the agent report.
