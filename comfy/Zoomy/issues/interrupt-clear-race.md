# `self._interrupt.clear()` can wipe a concurrently set interrupt

- Severity: low (race window; safe under Gradio's single worker).
- Status: verified open. `zoomy/local_engine.py:216` (`render_frame`),
  `:285` (`finalize_sequence`), `:172` (flag), `:1010-1013` (check).

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
