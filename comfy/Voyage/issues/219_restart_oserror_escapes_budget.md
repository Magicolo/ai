# 219 — `OSError` from `worker.restart()` escapes the restart budget (MEDIUM)

## Technical description
`_call_with_restart` (`supervisor.py:1042-1146`) gates only
`RecoverableWorkerError`. `worker.restart()` = `stop()` + `start()`;
`start()` raises `OSError` on `Popen` failure (`rpc.py:241`). That `OSError`
is not caught by either `except RecoverableWorkerError`, so it escapes raw,
bypassing `circuit_breaker_open` accounting and the
`FAILED`/`PAUSED_DISK_FULL` taxonomy (`run_segments` backstops it as generic
`FatalWorkerError`, losing the worker/op/budget context).

## Rationale
The budget promises "at most N restarts per worker per run, then
circuit-breaker". An `ENOENT`/`EACCES`/`ENOMEM` on respawn evades the counter
entirely (counter shows 1, no breaker event).

## Live evidence
Stub-driver probe in-image: stub `SubprocessWorker` with `stdin.write →
RecoverableWorkerError`, `restart → OSError(2)`, call
`sup._call_with_restart(w,'video','000000','health',{})` →
`OSError ESCAPED budget BUG: [Errno 2] No such file`, `restarts used:
{'video': 1}` with no `circuit_breaker_open` metric.

## Repro
As above; needs `tests/conftest.initialize_run_directory` +
`read_effective_config`.

## Source refs
- `Voyage/voyage/supervisor.py:1096-1143`
- `Voyage/voyage/rpc.py:230-246`

## Online sources
- https://man7.org/linux/man-pages/man2/flock.2.html — lock contention
  returns `EWOULDBLOCK` with `LOCK_NB`; not every `OSError` is contention
  (mirrors the narrow-catch discipline).
- https://www.man7.org/linux/man-pages/man2/fsync.2.html — `fsync` error
  taxonomy (`ENOSPC`/`EDQUOT`/`EIO`) must be branched, not string-matched.

## Fix candidates
Wrap `worker.restart()` + `restart_hook` in
`except (RecoverableWorkerError, OSError) as restart_exc` and route through
the same budget gate (mapping `OSError → RecoverableWorkerError` with a
`worker_restart_failed` event); keep `FatalWorkerError` from the hook
propagating.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation (2026-10-07)
Live probe in-container (`voyage:latest`): `OSError` absent from
`Supervisor._call_with_restart` source (`'OSError' in src == False`,
only `RecoverableWorkerError` gated). `rpc.SubprocessWorker.start`
raises `OSError` on `Popen` failure (`rpc.py:316-324`, fenced via
`except (VoyageError, OSError)`), and `restart()` is `stop()` +
`start()`, so a respawn `ENOENT`/`EACCES`/`ENOMEM` escapes the budget
gate raw — no `worker_restart_failed`, no `circuit_breaker_open`,
`run_segments` backstops it as generic `FatalWorkerError`. Confirmed
the issue as filed. Fix scope is `supervisor.py::_call_with_restart`
only (shared file: minimal hunk, unique anchors, no reformatting).

## Progress log
- Widened the inner `except RecoverableWorkerError` to
  `except (RecoverableWorkerError, OSError)` with an updated comment
  naming issue 219; `FatalWorkerError` from the hook still propagates
  (not caught). Updated the surrounding try comment to name the Popen
  `OSError` class.
- New tests in `tests/test_issue_219_restart_oserror.py` (3 tests):
  OSError restart trips the breaker with `worker_restart_failed` +
  `circuit_breaker_open`; OSError-then-success recovers; Fatal hook
  still propagates.
- Verified: `py_compile` OK; `ruff check` + `ruff format --check` clean;
  `mypy` strict clean on `supervisor.py`; scoped pytest 113 passed
  (219/231/243/260/263 + hardening/scoreboard/configure-extend/
  observability neighbors).

## Resolution (2026-10-07)
Fixed as proposed: `worker.restart()` + `restart_hook` `OSError`
routes through the same budget gate (consumes an attempt, emits
`worker_restart_failed`, re-enters the gate; terminal error stays
`FatalWorkerError` with `circuit_breaker_open`). No open items in
scope; outer `worker.call` `OSError` handling (if ever needed) stays out
of scope — `call()` only raises the worker taxonomy today.
