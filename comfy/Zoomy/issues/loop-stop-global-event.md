# Loop stop flag is global across sessions

- Severity: low (documented single-user assumption; breaks multi-user).
- Status: verified open, documented. `zoomy/rendering.py:48,82-99`,
  `zoomy/main.py:26`, `zoomy/interface.py` loop wiring.

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
