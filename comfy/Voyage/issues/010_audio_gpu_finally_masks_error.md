# 010 — `_with_audio_gpu` `finally` masks the real error and can strand video evicted

- Status: resolved in live tree (primary-error capture + best-effort teardown + metrics)
- Severity: HIGH (operability — masked error + stranded evict; resolved, record only)
- Group: correctness/GPU-swap — Rank: 2/5 (fixed)
- Area: correctness — GPU swap teardown (`voyage/supervisor.py`)
- Rank rationale: production-only path (acestep + streaming video) that hid the true
  failure and silently broke continuity for the next segment.

## Technical description

Pre-fix (`voyage/supervisor.py:736-753` at pass 1):

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
(a) If `generate_audio` raised (e.g. ACE OOM → Recoverable → budget exhausted →
`Fatal`), the `finally` ran two more restart-budgeted RPCs. If either raised, it
**replaced** the original exception — the operator saw `evict/rebuild` failure,
not the audio cause. If `recovery_path is None` (fake worker returns none; any
worker omitting it), `MediaError("no recovery tape")` masked the audio error.
(b) On the `recovery_path is None` path the video session was already evicted at
function entry and `rebuild` never ran — the worker was left evicted. The next
`generate_blocks` silently started a fresh stream (continuity break, no error).

Live state (re-verified 2026-09-30): `_with_audio_gpu`
(`voyage/supervisor.py:1011-1090`) captures the primary into `primary_error`,
runs teardown via `_best_effort_audio_teardown` (`:1093-1135`, log-never-raise,
`audio_swap_teardown_error` / `video_left_evicted` metrics), then re-raises the
primary; the no-tape case after success still raises `MediaError` but rebuild runs
even when the audio evict failed, so the stream is never stranded evicted silently.

## Why this is an issue

- Misattributed errors send debugging toward the video stack when audio failed.
- Silent continuity breaks defeat the project's core promise (see continuity work
  in DESIGN §22.5) with no metric or error marking the seam.
- Burns restart budget on teardown RPCs while a primary failure is in flight.

## Evidence

Live verification 2026-09-30:

```
$ rg -n "_with_audio_gpu|_best_effort_audio_teardown|no recovery tape" voyage/supervisor.py
1003:     def _with_audio_gpu(
1040:                 self._best_effort_audio_teardown(segment_id, recovery_path)
1060:                 raise MediaError(f"segment {segment_id}: no recovery tape for video rebuild")
1085:     def _best_effort_audio_teardown(self, segment_id: str, recovery_path: str | None) -> None:
```

Docstring contract (`:1017-1022`): "Teardown never masks the primary failure ...
the original exception propagates — including the no-tape case, where the video
is left evicted and marked explicitly."

## Reproduction

`acestep+longlive2`, force `generate_audio` failure with `recovery_tape=None`
(e.g. first segment with no tape, or fake-shaped `video` dict) → pre-fix
`MediaError: no recovery tape` surfaced instead of the audio error and video
stayed evicted; now the audio error propagates with a `video_left_evicted`
metric.

## Source references

- `voyage/supervisor.py:1003-1083` (`_with_audio_gpu`), `:1085-1131`
  (best-effort teardown), `:1133-` (`_ensure_audio_coverage` caller).

## Resolution candidates

1. (Landed) Capture the primary exception; run teardown best-effort (log, never
   raise when a primary is in flight); only raise `no recovery tape` when the try
   block succeeded.
2. (Landed) On teardown failure after success, attempt `rebuild` anyway or mark
   explicitly rather than leaving evicted residency implicit
   (`video_left_evicted` metric).
3. (Landed) Emit `audio_swap_teardown_error` so the failure is visible in
   `metrics.jsonl`/scoreboard (see 049).

## Online references

- Python `try/finally` semantics — "If the finally clause raises ... the original
  exception is lost" (hence capture-and-reraise):
  https://docs.python.org/3/tutorial/errors.html
- Python `BaseException` vs `Exception` (why the primary capture uses
  `BaseException`, incl. interrupts):
  https://docs.python.org/3/library/exceptions.html

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Resolution batch 3: primary capture + best-effort teardown + metrics landed.
- 2026-09-30: re-verified live (sites + contract present); reconstructed from
  archived pass-1 text (commit `b5d7dda`). Status → resolved.
