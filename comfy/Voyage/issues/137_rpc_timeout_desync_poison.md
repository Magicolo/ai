# 137 — RPC timeout leaves stale bytes that poison the next call (desync → Fatal id-mismatch)

- Severity: HIGH
- Area: correctness — worker RPC transport + supervisor timeout mix
- Files (as-read 2026-09-30):
  - `voyage/rpc.py:197-310` (`_read_response_line` + `call` + id-mismatch → Fatal)
  - `voyage/supervisor.py:766-770` (prefetch `decide` — no `timeout=` override)
  - `voyage/supervisor.py:592-599` (gauge `health` — `timeout=GAUGE_TIMEOUT_SECONDS`)
  - `voyage/supervisor.py:801-810` (`_embed_texts` — `timeout=EMBED_TIMEOUT_SECONDS`)
  - Context: `voyage/supervisor.py:296-325` (workers constructed with `timeout=config.voyage.rpc_timeout_seconds`, default 600)
- Not a duplicate of 001 (001 fixed the partial-line hang with a non-blocking reader + 8 MiB cap; this is what happens *after* that reader correctly times out).

## Technical description

`SubprocessWorker.call` (`voyage/rpc.py:261-315`) sends one JSONL request (`req-NNNNNN`), then
`_read_response_line` (`voyage/rpc.py:197-259`) accumulates bytes on a non-blocking fd until `\n`,
the 8 MiB cap (`MAX_RESPONSE_LINE_BYTES = 8 * 1024 * 1024`, `voyage/rpc.py:51`), or the deadline.
On expiry it raises `RecoverableWorkerError` — correctly — but **leaves the late response in the
pipe**. Nothing drains the fd, discards the in-flight request, or fences the worker.

The next `call` writes `req-(N+1)` and reads whatever arrives first, which is the *late* `req-N`
response. The id check then fires:

```python
# voyage/rpc.py:303-304 (as-read)
if response.id != request.id:
    raise FatalWorkerError(f"worker {self._module} id mismatch: {response.id}")
```

So a transient slow worker (one timeout) is converted into a **Fatal** on the *next* call —
outside the restart budget (`_call_with_restart` only retries `RecoverableWorkerError`,
`voyage/supervisor.py:472-528`). The supervisor mixes four timeout domains on the same pipes,
which makes the desync reachable from several paths:

- default `config.voyage.rpc_timeout_seconds` (600 s) for generate/benchmark/resume ops
  (`voyage/supervisor.py:296-325`);
- `GAUGE_TIMEOUT_SECONDS = 5.0` (`voyage/supervisor.py:114`) for `health` (`:592-599`);
- `EMBED_TIMEOUT_SECONDS = 60.0` (`voyage/supervisor.py:120`) for `embed` (`:801-810`);
- prefetch `decide` with **no** override (`voyage/supervisor.py:766-770` → `self._director.call("decide", payload)`),
  so a ~5 min CPU Qwen decide holds the prefetch thread at the full 600 s default (see also issue 030).

Any op that is slow-but-alive (not dead) can dribble its response *after* the caller gave up,
and the following op on that worker eats it.

## Why this is an issue

- Turns a recoverable slowness into a non-retryable Fatal, defeating the Phase 6 slice B
  restart budget / circuit-breaker design.
- The Fatal message (`id mismatch`) misdirects diagnosis toward a protocol bug instead of
  a timeout hangover.
- Mixed timeouts widen the window: a 5 s gauge timeout against a worker busy with a 60 s+
  op almost guarantees a stale line for the next caller.

## Live evidence

Live re-verification 2026-09-30 (read-only; probes per task brief ran in `voyage:latest` CPU-only).
Track A draft command+output bundle (`ses_f0fbea412ffeh3V7K1SlQpXjsv`) was not recoverable from
this writer's context, so evidence below is the as-read code, not invented command output:

```
$ sed -n '197,315p' voyage/rpc.py
    def _read_response_line(...):  # 197 — non-blocking os.read until \n / cap / deadline
    ...
    def call(...):                 # 261 — write req-N, _read_response_line, decode_response
    ...
    if response.id != request.id:  # 303
        raise FatalWorkerError(...)  # 304 — Fatal, not Recoverable

$ sed -n '766,770p;592,599p;801,810p' voyage/supervisor.py
    766:        def _call() -> dict[str, Any] | None:
    767:            try:
    768:                return self._director.call("decide", payload)  # no timeout=
    594:                    health = worker.call("health", {}, timeout=GAUGE_TIMEOUT_SECONDS)  # 5.0
    808:            result = self._director.call("embed", {"texts": texts}, timeout=EMBED_TIMEOUT_SECONDS)  # 60.0
```

As-read values: `MAX_RESPONSE_LINE_BYTES = 8 * 1024 * 1024` (`rpc.py:51`),
`RESPONSE_READ_CHUNK_BYTES = 65536` (`rpc.py:57`), `GAUGE_TIMEOUT_SECONDS = 5.0`,
`EMBED_TIMEOUT_SECONDS = 60.0`, `rpc_timeout_seconds = 600.0` (`config.py:458`).

## Minimal repro

1. Start a fake worker; stub its handler to sleep 2 s then reply (slow-but-alive).
2. `worker.call("health", {}, timeout=0.2)` → `RecoverableWorkerError` at ~0.2 s (correct).
3. Immediately `worker.call("health", {}, timeout=5)` → gets the *first* call's late reply;
   `response.id (req-000001) != request.id (req-000002)` → `FatalWorkerError: id mismatch`.
4. Same shape via supervisor: gauge `health` (5 s) racing a busy director, then the next
   `decide`/`embed` eats the stale line.

## Fix candidates

1. (Preferred) Fence after timeout: on `_read_response_line` expiry, `stop()`/restart the worker
   before raising `RecoverableWorkerError`, so no stale bytes survive (timeout ≡ restart, like
   the commit path already assumes). Cost: one restart per timeout; matches the existing
   `_call_with_restart` contract.
2. Drain-with-deadline: after timeout, non-blocking drain to `\n` with a short grace (e.g. 1 s);
   on success discard + retry once, else restart. Avoids restarts for near-miss slowness but adds
   a second deadline path to test.
3. Generation fencing: tag each request with a monotonic generation; ignore lines whose id does
   not match the outstanding request (drain until match or deadline). Keeps the worker alive but
   requires the reader to distinguish late-vs-garbage lines.
4. Unify timeouts per worker or document the mix: at minimum give the prefetch `decide` an
   explicit short budget (issue 030's `PREFETCH_TIMEOUT_SECONDS` proposal) so the 600 s default
   never parks a thread behind a stale reply.
5. Regression tests: slow-then-reply stub → timeout → next call succeeds (no Fatal); mixed-timeout
   sequence (5 s health vs busy worker) → no desync; prefetch-timeout → miss, not poisoned commit.

## References

- In-tree: `voyage/rpc.py:51,57,197-315`; `voyage/supervisor.py:114,120,296-325,472-528,592-599,766-770,801-810`;
  `voyage/config.py:458,478`; `voyage/workers/loop.py` (framing counterpart).
- Neighbor issues: 001 (deadline-bypass fix this builds on), 014 (restart bypasses budget),
  017 (gauge/embed timeouts), 030 (prefetch has no timeout), 100 (NaN timeout validation).
- External:
  - https://docs.python.org/3/library/select.html (readiness ≠ complete line)
  - https://docs.python.org/3/library/os.html#os.set_blocking
  - https://jsonlines.org/ (one JSON value per `\n`-terminated line)

## Investigation log

- 2026-09-30: filed by Track A sweep (6 new issues); live re-verified via Read (concurrent
  uncommitted edits noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`,
  `config/persistence/rpc/supervisor` — all citations are as-read values above).
