# `self._interrupt.clear()` can wipe a concurrently set interrupt

- Severity: low (race window; safe under Gradio's single worker).
- Status: FIXED. The `threading.Event` is gone, replaced by
  consume-on-observe under a lock: `request_interrupt` sets
  `_interrupt_pending`, `_begin_job` takes ownership into
  `_abort_job_at_start`, and `_raise_if_interrupted` aborts once while
  clearing both flags — so a press racing job start is never wiped, an
  idle press aborts the next job at its first checkpoint, and one press
  aborts exactly one job. Tests: idle-abort, consumed-never-kills-next,
  and generator-level idle abort (the old Event-pinning test was replaced —
  it asserted the removed mechanism).

## Evidence

Both entry points call `self._interrupt.clear()` first thing. The UI
thread's Interrupt button calls `request_interrupt()` → `self._interrupt.set()`
(`local_engine.py:179-180`). If the set lands between a job's start and
its `clear()`, the request is silently discarded and the job runs to
completion despite the user pressing Interrupt. No lock protects the flag
or the `_frame_pipeline/_rife_model/_effects_stack/_music_stack` caches
(`:176-179` area) either — safe only under the single-worker assumption
(`main.py:26`), not as a library.

## Fix

Clear-then-check under a lock, or consume the flag with an atomic
test-and-clear at the first `_raise_if_interrupted` checkpoint instead of
an unconditional clear at entry (a stale set from a *previous* job must
still not kill the next job — that is what the clear is for — so the
correct primitive is "clear only interrupts predating this job", e.g. a
per-job generation counter). Document the chosen semantics.

## Verification

- New test: set-then-start ordering never loses the interrupt; stale set
  from a prior job never kills the next job.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
