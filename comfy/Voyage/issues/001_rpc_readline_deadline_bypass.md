# 001 — RPC `readline()` blocks past the select deadline on partial lines (supervisor can hang indefinitely)

- Status: resolved in live tree (fix landed — non-blocking `os.read` reader + 8 MiB cap)
- Severity: HIGH (liveness — hang; resolved, record only)
- Group: correctness/RPC — Rank: 1/5 (critical pattern, fixed)
- Area: correctness — worker RPC transport (`voyage/rpc.py`)
- Rank rationale: defeated the only timeout guard on every worker call; one sick worker wedged commits forever.

## Technical description

The pre-fix `SubprocessWorker.call` (`voyage/rpc.py:164-179` at pass 1) implemented
the call timeout as a `select.select()` deadline on readability, then called the
**blocking** `proc.stdout.readline()`:

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

`select` only guarantees *some* bytes are readable, not a full `\n`-terminated
line. A worker that dribbles a partial line (OOM loop printing without newline,
upstream model code leaking to stdout despite `redirect_stdout`, half-killed
process holding the pipe open) makes `select` return ready immediately, then
`readline()` blocks **past `deadline` forever**. The old docstring conceded it
(`voyage/rpc.py:148-150` at pass 1):

> `a worker dribbling a partial line forever would still block readline (not a real
> worker behavior: loop.py writes the full line at once)`

"Not a real worker behavior" is exactly the assumption a fault-injection /
crash-matrix testbed must not make. The caller also holds `self._call_lock`, so a
wedged call could block the director-prefetch thread and the commit path behind
it (see 033).

Live state (re-verified 2026-09-30): the reader is now `_read_response_line`
(`voyage/rpc.py:195-268`) — fd set non-blocking via `os.set_blocking`, accumulated
with `os.read` until `\n`, cap, or deadline; never a blocking `readline` after
`select`. Hard cap `MAX_RESPONSE_LINE_BYTES = 8 * 1024 * 1024` (`voyage/rpc.py:51`).

## Why this is an issue

- `rpc_timeout_seconds` (default 600) is documented as the bound on worker calls;
  pre-fix it was unenforced in the partial-line case.
- No restart, no circuit breaker fired — the supervisor thread simply never returned.
- `run_segments` has no outer watchdog, so the whole run wedged with status RUNNING.

## Evidence

Live verification 2026-09-30 (fix present):

```
$ rg -n "os\.set_blocking|MAX_RESPONSE_LINE_BYTES|readline|os\.read" voyage/rpc.py
51:MAX_RESPONSE_LINE_BYTES = 8 * 1024 * 1024
53:#: Bytes per os.read while accumulating a response line. Small enough to
197:        Never calls blocking `readline()` after `select`: the fd goes
209:        # TextIO. Both select and os.read accept either form.
218:        # a single os.read still returns promptly even if the flag
222:            os.set_blocking(raw_fd, False)
237:                chunk = os.read(raw_fd, RESPONSE_READ_CHUNK_BYTES)
247:            if len(buffer) > MAX_RESPONSE_LINE_BYTES:
250:                    f"{MAX_RESPONSE_LINE_BYTES} bytes on {op}"
268:        deadline. Lines past MAX_RESPONSE_LINE_BYTES fail the same way.
```

Pass-1 evidence (archived): `rg -n "readline|deadline" voyage/rpc.py` showed
`line = proc.stdout.readline()` at line 176 with the `while True` + `break` never
re-checking the deadline during the read.

## Reproduction

1. Start any `SubprocessWorker` (e.g. fake video worker).
2. Make the worker write `b"partial..."` with no `\n` and sleep (stub `handle_health`
   to `os.write(stdout, b"junk")` without newline).
3. Call `worker.call("health", {})` with `timeout=5`.
4. Pre-fix observed: `select` returns at once; `readline()` blocks past 5 s
   indefinitely. Expected (now actual): `RecoverableWorkerError` at ~5 s followed
   by a worker restart.

## Source references

- `voyage/rpc.py:48-57` (cap constants), `:195-268` (bounded reader), `:313`
  (health probe); `voyage/workers/loop.py:50-130` (framing counterpart).
- Pre-fix sites (pass 1): `voyage/rpc.py:148-150` (admission), `:164-179` (loop),
  `:90-94` (`_call_lock`).

## Resolution candidates

1. (Landed) Never call blocking `readline` after `select`: set the fd
   non-blocking (`os.set_blocking(False)`) and accumulate with `os.read` until `\n`
   or deadline; on expiry `restart()` the worker, then raise
   `RecoverableWorkerError`. Keeps the framing, bounds the wait.
2. (Alternative, not taken) Run `readline` in a helper thread with
   `join(deadline)`; on timeout abandon the fd, restart, raise. Simpler but leaks
   a blocked thread per incident.
3. (Landed with 1) Length-cap the line buffer (a worker sending GBs without
   newline is the same DoS with a newline at the end) — 8 MiB cap.
4. Regression test: stub worker that writes a partial line and sleeps; assert
   `call(..., timeout=0.2)` raises within ~1 s and the worker is restarted.

## Online references

- Python `select.select` — "The return value is a triple of lists of objects that
  are ready" (readiness ≠ complete line):
  https://docs.python.org/3/library/select.html
- Python `os.set_blocking` — "Set blocking or non-blocking mode of the specified
  file descriptor":
  https://docs.python.org/3/library/os.html#os.set_blocking
- JSON Lines framing (one JSON value per `\n`-terminated line):
  https://jsonlines.org/

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep (pass 1); admission quote verified via `rg`.
- Resolution batch 3: non-blocking reader + 8 MiB cap landed (`rpc.py`).
- 2026-09-30: re-verified live (reader + cap present, no blocking `readline` on the
  call path); reconstructed from archived pass-1 text (commit `b5d7dda`) with fresh
  live evidence. Status → resolved.
