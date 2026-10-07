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
