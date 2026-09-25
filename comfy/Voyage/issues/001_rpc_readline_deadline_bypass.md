# 001 — RPC `readline()` blocks past the select deadline on partial lines (supervisor can hang indefinitely)

- Status: resolved 2026-09-25 (supervisor track)
- Severity: critical (hang / liveness)
- Area: correctness — worker RPC transport
- Rank rationale: defeats the only timeout guard on every worker call; one sick worker wedges commits forever.

## Technical description

`voyage/rpc.py:164-179` implements the call timeout as a `select.select()` deadline on
readability, then calls the **blocking** `proc.stdout.readline()`:

```python
deadline = time.monotonic() + effective_timeout
while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RecoverableWorkerError(...)
    ready, _, _ = select.select([proc.stdout], [], [], remaining)
    if not ready:
        raise RecoverableWorkerError(...)
    line = proc.stdout.readline()   # <-- unbounded, no timeout
```

`select` only guarantees *some* bytes are readable, not a full `\n`-terminated line.
A worker that dribbles a partial line (OOM loop printing without newline, upstream
model code leaking to stdout despite `redirect_stdout`, half-killed process holding
the pipe open) makes `select` return ready immediately, then `readline()` blocks
**past `deadline` forever**. The docstring at `voyage/rpc.py:148-150` concedes it:

> `a worker dribbling a partial line forever would still block readline (not a real
> worker behavior: loop.py writes the full line at once)`

"Not a real worker behavior" is exactly the assumption a fault-injection/crash-matrix
testbed must not make. The caller also holds `self._call_lock`, so a wedged call can
block the director-prefetch thread and the commit path behind it (see 033).

## Why this is an issue

- `rpc_timeout_seconds` (default 600) is documented as the bound on worker calls; it
  is unenforced in the partial-line case.
- No restart, no circuit breaker fires — the supervisor thread simply never returns.
- `run_segments` has no outer watchdog, so the whole run wedges with status RUNNING.

## Evidence

Read-only source inspection (no GPU needed):

```
$ rg -n "readline|deadline" Voyage/voyage/rpc.py
148:        wait is a select-deadline on readability — a worker dribbling a
149:        partial line forever would still block `readline` (not a real
164:            deadline = time.monotonic() + effective_timeout
166:                remaining = deadline - time.monotonic()
176:                line = proc.stdout.readline()
```

The `while True` + `break` never re-checks the deadline during the read.

## Reproduction

1. Start any `SubprocessWorker` (e.g. fake video worker).
2. Make the worker write `b"partial..."` with no `\n` and sleep (stub `handle_health`
   to `os.write(stdout, b"junk")` without newline).
3. Call `worker.call("health", {})` with `timeout=5`.
4. Observed: `select` returns at once; `readline()` blocks past 5 s indefinitely.
   Expected: `RecoverableWorkerError` at ~5 s followed by a worker restart.

## Source references

- `voyage/rpc.py:148-150` (admission), `:164-179` (loop), `:90-94` (`_call_lock`).
- `voyage/supervisor.py:183-191` (single `ThreadPoolExecutor(1)` sharing the lock —
  see 033).

## Resolution candidates

1. (Preferred) Never call blocking `readline` after `select`: set the fd
   non-blocking (`os.set_blocking(False)`) and accumulate with `os.read` until `\n`
   or deadline; on expiry `restart()` the worker, then raise
   `RecoverableWorkerError`. Keeps the framing, bounds the wait.
2. Run `readline` in a helper thread with `join(deadline)`; on timeout abandon the
   fd, restart the worker, raise. Simpler but leaks a blocked thread per incident.
3. Length-cap the line buffer as well (a worker sending GBs without newline is the
   same DoS with a newline at the end).

Add a regression test: stub worker that writes a partial line and sleeps; assert
`call(..., timeout=0.2)` raises within ~1 s and the worker is restarted.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep (sub-agent), admission quote verified by
  orchestrator via `rg`. No fix attempted (investigation-only pass).
- Open: implement (1), add test, re-run `test_failure_policy.py` + crash matrix.
- 2026-09-25 (repair pass): refs verified current (`rpc.py:148-150,164-179,94`;
  `supervisor.py:183-191`); `## Why this is an issue` already present, no change.
- 2026-09-25 (RESOLVED, supervisor track): implemented candidate 1 (+3).
  `SubprocessWorker.call` (`voyage/rpc.py:236-265`) no longer calls blocking
  `readline()` after `select`: new `_read_response_line`
  (`voyage/rpc.py:169-233`) sets the fd non-blocking and accumulates with
  `os.read` until `\n`, EOF, the 8 MiB `MAX_RESPONSE_LINE_BYTES` cap
  (`voyage/rpc.py:38`), or the deadline — every exit except a full line
  raises `RecoverableWorkerError`, so the restart path engages and the
  `_call_lock` is always released. Malformed response bytes now also raise
  `RecoverableWorkerError` instead of escaping as pydantic `ValidationError`
  (`voyage/rpc.py:254-261`). Raw-fd test doubles (int stdout, as in
  `test_failure_policy.py`) are accepted alongside `TextIO`. Regression
  tests in `Voyage/tests/test_commit_hardening.py`: partial-line stub fails
  at ~0.5 s (`test_partial_line_never_passes_deadline`), full line still
  reads (`test_full_response_line_still_reads`), oversize line fails fast at
  the cap (`test_oversize_response_line_fails_fast`). Gates: full
  `Voyage/scripts/gates.sh` green (626 passed).
