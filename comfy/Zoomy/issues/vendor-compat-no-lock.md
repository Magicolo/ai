# `_APPLIED_PATCHES` mutated without a lock

- Severity: low (race on concurrent first-music-render; single-user
  service in practice).
- Status: verified open. `zoomy/vendor_compat.py:30,44-60,106`.

## Evidence

`apply_transformers5_compat` checks `if _CPU_INIT_PATCH in _APPLIED_PATCHES`
then patches and adds — check-then-act with no lock. Two threads
rendering music for the first time concurrently can both pass the check
and double-wrap `PreTrainedModel.get_init_context` (the second wrap
captures the already-wrapped bound method; the inner
`__zoomy_no_meta__` guard at `:58` catches the *same-thread re-entry*
case, not the interleaved one). Gradio runs with
`default_concurrency_limit=1` (`main.py:26`), so this needs a library-use
or future concurrency bump to bite.

## Fix

Guard with a module-level `threading.Lock` (double-checked pattern), or
make application idempotent by keying on the
`__zoomy_no_meta__` attribute alone. Test with two threads racing first
call: `get_init_context` wrapped exactly once.

## Verification

- New test: concurrent first calls → single wrap, no exception.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
