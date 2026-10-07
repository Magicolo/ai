# 223 — Control-plane `read-then-write` race survives `_write_state_preserving_control_plane` (MEDIUM)

## Technical description
`supervisor.py:2386-2403` re-reads `state.json` to honor
`STOP/PAUSE_REQUESTED` written lock-free by `voyage stop/pause`, then
`write_state`. A request landing **after** the live read but **before**
`write_state` is still clobbered. The window is microseconds (one
`read_state` + one atomic write), but the commit path holds it on *every*
segment, and `test_recovery.py:67-89` already documents the sibling race
("pre-existing race … silently clobbered").

```python
try: live_status = read_state(...).status / except VoyageError: live_status = fresh.status
if live_status in (...): fresh.status = live_status
write_state(self._run_dir, fresh)
```
No lock, no `O_EXCL`/CAS — two writers (`commit` under `_held_run_lock`,
`stop/pause` without it) interleave by design.

## Rationale
`stop`/`pause` is the only operator brake on an infinite run. Losing it
means the run sails past the boundary it was told to rest at; the next
boundary re-reads, so the delay is one segment — but on 4-minute LTX
segments that is 4 minutes of unwanted GPU burn.

## Live evidence
Code shape (race is structural, not probabilistic in unit tests) — source
printed via `inspect.getsource`, see above.

## Repro
Thread A loops `_write_state_preserving_control_plane`, thread B writes
`STOP_REQUESTED` between A's `read_state` return and `write_state` entry
(inject via monkeypatched `read_state` barrier); assert B's request is lost.

## Source refs
- `Voyage/voyage/supervisor.py:2386-2403,2612-2624,2707-2793`
- `Voyage/tests/test_recovery.py:73-80` comment

## Online sources
- https://pkg.go.dev/github.com/larsartmann/go-atomic-write%40v0.3.0 —
  fingerprint + lock + verify (read-verify-write, not read-then-write).
- https://man7.org/linux/man-pages/man2/flock.2.html — advisory lock held
  across the whole read-modify-write, not just the media section.

## Fix candidates
1. Hold `_held_run_lock` for control-plane writes too (stop/pause take the
   lock non-blocking, fail with "commit in flight, retry").
2. Or add a `control_plane.json` sidecar the commit path merges instead of
   overwriting.
3. At minimum document the one-segment overrun in DESIGN §73.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.
