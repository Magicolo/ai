# 010 — `_with_audio_gpu` `finally` masks the real error and can strand video evicted

- Status: resolved 2026-09-25 (supervisor track)
- Severity: major (operability / continuity)
- Area: correctness — GPU swap teardown (`voyage/supervisor.py:736-753`)
- Rank rationale: production-only path (acestep + streaming video) that hides the
  true failure and silently breaks continuity for the next segment.

## Technical description

```python
if swap:
    self._call_with_restart(self._video, "video", segment_id, "evict_gpu", {})
try:
    return self._call_with_restart(self._audio, "audio", segment_id, "generate_audio", audio_payload)
finally:
    if swap:
        self._call_with_restart(self._audio, "audio", segment_id, "evict_gpu", {})
        if recovery_path is None:
            raise MediaError(f"segment {segment_id}: no recovery tape for video rebuild")
        self._call_with_restart(self._video, "video", segment_id, "rebuild", {"recovery_path": recovery_path})
```

Two defects:
(a) If `generate_audio` raises (e.g. ACE OOM → Recoverable → budget exhausted →
`Fatal`), the `finally` runs two more restart-budgeted RPCs. If either raises, it
**replaces** the original exception — the operator sees `evict/rebuild` failure,
not the audio cause. If `recovery_path is None` (fake worker returns none; any
worker omitting it), `MediaError("no recovery tape")` masks the audio error.
(b) On the `recovery_path is None` path the video session was already evicted at
function entry and `rebuild` never runs — the worker is left evicted. The next
`generate_blocks` silently starts a fresh stream (continuity break, no error).

## Why this is an issue

- Misattributed errors send debugging toward the video stack when audio failed.
- Silent continuity breaks defeat the project's core promise (see continuity work
  in DESIGN §22.5) with no metric or error marking the seam.
- Burns restart budget on teardown RPCs while a primary failure is in flight.

## Evidence

Source quotes above; swap triggers exactly for `acestep+streaming`, i.e. production
GPU runs (`supervisor.py:733`).

Re-verified 2026-09-25:

```
$ rg -n "evict_gpu|rebuild" voyage/supervisor.py
737:            self._call_with_restart(self._video, "video", segment_id, "evict_gpu", {})
744:                self._call_with_restart(self._audio, "audio", segment_id, "evict_gpu", {})
746:                    raise MediaError(f"segment {segment_id}: no recovery tape for video rebuild")
751:                    "rebuild",
```

## Reproduction

`acestep+longlive2`, force `generate_audio` failure with `recovery_tape=None`
(e.g. first segment with no tape, or fake-shaped `video` dict) →
`MediaError: no recovery tape` surfaces instead of the audio error; video stays evicted.

## Source references

- `voyage/supervisor.py:718-753` (`_with_audio_gpu`, `_ensure_audio_coverage`).

## Resolution candidates

1. Capture the primary exception; run teardown best-effort (log, never raise when
   a primary is in flight — `sys.exception()` guard); only raise `no recovery
   tape` when the try block succeeded.
2. On teardown failure after success, attempt `rebuild` anyway or mark the run
   FAILED explicitly rather than leaving evicted residency implicit.
3. Emit a metric (`audio_swap_teardown_error`) so the failure is visible in
   `metrics.jsonl`/scoreboard (see 049).

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Open: implement + test (failing-audio + missing-tape matrix).
- 2026-09-25 (repair pass): refs verified current (`supervisor.py:732-751`);
  Evidence enriched with live `rg` output; `## Why this is an issue` already
  present, no change.
- 2026-09-25 (RESOLVED, supervisor track): implemented candidates 1–3. The
  `finally` is gone: `_with_audio_gpu` (`voyage/supervisor.py:862-942`)
  captures the primary exception, runs teardown through
  `_best_effort_audio_teardown` (`voyage/supervisor.py:944-983`, log-only,
  never raises while a primary is in flight), and re-raises the primary
  unchanged — the `no recovery tape` `MediaError` now fires only when the
  take render SUCCEEDED, and the no-tape failure case is recorded as an
  explicit `video_left_evicted` metric (the run still rests FAILED via the
  propagating primary, so the seam is marked, not silent). After success,
  an audio-evict failure still attempts the video rebuild before raising,
  and every teardown failure emits `audio_swap_teardown_error` (candidate
  3, visible in metrics/scoreboard readers). Matrix tests in
  `Voyage/tests/test_commit_hardening.py` (stubbed swap, no GPU):
  `test_audio_failure_propagates_not_masked` (ACE OOM, no tape — cause
  preserved, evict-only calls, `video_left_evicted` logged),
  `test_audio_failure_with_tape_still_rebuilds` (rebuild attempted, cause
  still wins), `test_success_path_teardown_failure_still_rebuilds`
  (rebuild attempted, teardown error raised + metric). Gates: full
  `Voyage/scripts/gates.sh` green (626 passed).
