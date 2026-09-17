# `find_family` failures escape UI handlers as tracebacks

- Severity: medium (validation + info disclosure via `show_error=True`).
- Status: FIXED. `find_family` now sits inside a guarded region in all
  four handlers, each returning/yielding its native `**Error:**` shape on
  an unknown key (offline badge + error line for status; cleared-preview
  error tuple; single error yield reusing the finalize error path; refused
  delete for clear). Tests submit `no-such-family` to every handler
  (`test_*_reports_unknown_family`, via a `_wiring_context` helper that
  builds `InterfaceContext` over stub components). `_draw_family_panel`
  still raises at build time — correct fail-fast for a programming error.
  `zoomy/interface.py` (`_bind_refresh_status/_bind_refresh_previews/_bind_finalize/_bind_clear_frames`).

## Evidence

`find_family` raises `ZoomyError` on an unknown key. Four bind handlers
call it *outside* any `try`:

- `_bind_refresh_status.refresh_status` (`interface.py:309`)
- `_bind_refresh_previews.refresh_previews` (`interface.py:330`)
- `_bind_finalize.finalize_sequence` (`interface.py:372` — before the `try`
  at `:379`)
- `_bind_clear_frames.clear_frames` (`interface.py:416`)

plus `_draw_family_panel` (`:474`, build time). Only the interrupt path
catches `ZoomyError`. With `show_error=True` (`main.py:31`), a crafted
dropdown payload surfaces a server traceback — including absolute
container paths — to the client. Trigger is low-severity today (Gradio
dropdowns submit known keys), but the timer-driven `refresh_status` runs
every 10 s, so one bad state poisons the stats line permanently.

## Fix

Move the `find_family` call inside each handler's `try` (or wrap the body)
and return/yield the same `**Error:** …` message shape the finalize path
already uses. Add handler-level tests submitting an unknown family key to
`refresh_status`, `refresh_previews`, `finalize_sequence`, and
`clear_frames`, asserting a clean error string and no raise.

## Verification

- New tests: unknown `family_key` → error message, no exception, for all
  four handlers.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
