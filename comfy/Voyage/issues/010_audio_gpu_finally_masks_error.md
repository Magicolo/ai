# 010 — `_with_audio_gpu` `finally` masks the real error and can strand video evicted

- Status: open
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
