# 058 — DONE commit is two-step non-atomic; stderr log fd leaks; `stop()` can leave zombies (+ `*.tmp.npy` orphan blind spot)

- Status: resolved (fixed 2026-09-25)
- Severity: low-medium (durability nits + fd/zombie leaks + invisible torn-vector
  orphans)
- Area: `voyage/supervisor.py:1219-1222`, `voyage/rpc.py:96-122`,
  `voyage/concepts.py:158`
- Rank rationale: each small; together they make `validate`'s orphan scan
  unreliable and leak resources per restart.

## Technical description

```python
done_partial = segment / "DONE.partial"
atomic_write_bytes(done_partial, b"")   # temp+fsync+replace to "DONE.partial" + fsync_dir
done_partial.replace(segment / paths.DONE_MARKER)  # plain rename, then fsync_dir
```

`atomic_write_bytes(DONE.partial)` already leaves a visible `DONE.partial`; the
second rename adds a window where concurrent `validate_run`
(`rglob("*.partial")`) reports `orphan partial files` on a healthy in-flight
commit. Direct `atomic_write_bytes(DONE)` (mkstemp + `os.replace`) would never
expose the intermediate. Related: concept-vector temps
`concept_vectors.npy.<pid>.tmp.npy` (`concepts.py:158`) never match `*.partial`,
so torn-vector orphans are invisible to the scan. Per-`start()` log fd
(`rpc.py:98`) is never closed (3 workers × restarts = leaked fds); `stop()` never
closes `stdout`/log handles and never `wait()`s after `kill()` (`rpc.py:119-122`)
→ zombies.

## Why this is an issue

Each item here is small, but together they undermine the durability story
the supervisor promises: a spurious "orphan partial files" report during a
healthy commit sends operators chasing corruption that isn't there; leaked
log fds accumulate across worker restarts on long runs until the supervisor
hits fd limits; and an unreaped zombie per killed worker confuses process
supervision. The invisible `*.tmp.npy` orphans are the sharpest edge — torn
concept vectors pass `validate` silently and then poison novelty scoring
(see 059). Individually one-line fixes; collectively they make `validate`
trustworthy.

## Evidence

```
$ sed -n '1219,1222p' Voyage/voyage/supervisor.py
            done_partial = segment / "DONE.partial"
            atomic_write_bytes(done_partial, b"")
            done_partial.replace(segment / paths.DONE_MARKER)
            fsync_dir(segment)
$ sed -n '113,124p' Voyage/voyage/rpc.py   # stop(): waits BEFORE kill, never after
    def stop(self) -> None:
        proc, self._proc = self._proc, None
        ...
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()                     # no wait() after kill → zombie
$ sed -n '158,168p' Voyage/voyage/concepts.py
        tmp_npy = self._vectors_path.parent / f"{self._vectors_path.name}.{os.getpid()}.tmp.npy"
```

Source quotes above (all re-verified live 2026-09-25).

## Reproduction

Concurrent `validate_run` during a commit (spurious orphan report); restart
workers N times and count open fds / zombie processes; kill mid-vector-append and
run the orphan scan (misses `*.tmp.npy`).

## Source references

- Files/lines above.

## Resolution candidates

Write DONE via one `atomic_write_bytes(segment/DONE, b"")`; extend the orphan
scan to `*.tmp.npy`/`*.tmp*`; keep a `self._log_file` handle, close on `stop()`;
after `kill()` do `proc.wait()`.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- 2026-09-25 (repair): re-verified all three sites live (Evidence pasted).
  Partial improvement noted: `stop()` now does `proc.wait(timeout=10)` before
  `kill()` (`rpc.py:119-122`) — but still never `wait()`s AFTER `kill()`, and
  the per-`start()` log fd (`rpc.py:99-101`) is still never closed. Claims
  stand. Added `## Why this is an issue`.
- Open: implement + tests.
- 2026-09-25 (consumer-side orphan-scan half — FIXED): `cli.validate_run`
  scans `_ORPHAN_PATTERNS = ("*.partial", "*.tmp.npy", "*.tmp*")` under
  `segments/` plus the `novelty/` dir (torn `concept_vectors.npy.*.tmp.npy`
  now visible), deduplicated, reported as `orphan transient files`
  (keeps the `orphan` substring existing tests match). Tests in
  `tests/test_run_relative_consumer.py` (segment + novelty tmp orphans
  flagged, clean run green). DONE-atomicity + rpc fd/zombie halves remain
  with the supervisor track (supervisor.py/rpc.py out of scope).
- 2026-09-25 (supervisor halves — DONE, status untouched for the media
  track): single-step DONE — `atomic_write_bytes(segment / DONE, b"")`
  directly (`voyage/supervisor.py:1456-1461`), no more visible
  `DONE.partial` window for concurrent validate to trip on (the mkstemp
  `DONE.*.partial` temp still exists mid-write, same as every other
  atomic file — the scan flags real remnants, and
  `test_validate_detects_done_partial_remnant` still passes on a crafted
  remnant). Log-fd lifecycle (`voyage/rpc.py:107-153`): `start()` keeps
  `self._log_file` (closing any previous handle first) and `stop()`
  always closes it, including the never-started path. Zombie reap
  (`voyage/rpc.py:130-153`): `stop()` now `wait()`s after `kill()` (10 s
  grace, then gives up — SIGKILL landing is near-instant, so this cannot
  block). Tests in `Voyage/tests/test_commit_hardening.py`:
  `test_done_written_without_partial_remnant` (DONE present, zero
  `*.partial` under `segments/`) and
  `test_worker_stop_closes_log_and_reaps` (SIGKILL → `stop_workers`
  completes, log handle closed, `_proc`/`_log_file` cleared). The
  `concepts.py` tmp-scan half stays with the media track. Gates: full
  `Voyage/scripts/gates.sh` green (626 passed).
- 2026-09-25 (review): both halves verified landed — single-step
  `atomic_write_bytes(segment/DONE)`, `SubprocessWorker` log-fd lifecycle
  (`_log_file` closed on every `stop()`, `wait()` after `kill()`), orphan scan
  extended to `*.tmp.npy`/`*.tmp*` under `segments/` and `novelty/`
  (`cli._ORPHAN_PATTERNS`). Tests: no-`DONE.partial`, kill→stop reaps with fds
  closed, tmp-orphan flags. Issue fully resolved.
