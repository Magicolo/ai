# Loop stop flag is global across sessions

- Severity: low (documented single-user assumption; breaks multi-user).
- Status: FIXED. `render_loop`/`generate_video` take an optional
  keyword-only `stop_event` (resolved via `_resolve_stop_event`, defaulting
  to the single-session global, so the interface is untouched and behavior
  is unchanged). Test: `test_concurrent_loops_stop_independently` drives
  two loops on separate flags — stopping one leaves the other rendering.

## Evidence

`_loop_stop_requested = threading.Event()` is module-level: one user's
stop/uncheck stops every running loop. The docstring on
`request_loop_stop` (`rendering.py:82-89`) now states the assumption
explicitly ("one global flag is sufficient because zoomy serves a single
local user"), and `main.py:26` pins `default_concurrency_limit=1`, which
makes it true *today*. The sibling `LocalEngine._interrupt`
(`local_engine.py:172`) is per-instance — inconsistent granularity that
will confuse the first person to scope one and not the other.

## Fix (when sessions matter; NOT today)

Scope the flag to the loop invocation (pass an `Event` through
`render_loop`/`generate_video`, owned by the calling session), keeping
the module-level helpers as thin wrappers for the single-session UI. Do
not change behavior in this fix — add the seam plus a test proving two
concurrent loops stop independently.

## Verification

- New test: two loop instances, stop one → the other continues.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
