# 004 — No inter-process mutual exclusion: two supervisors corrupt the same segment + state

- Status: resolved in live tree (`fcntl` single-writer lock landed)
- Severity: HIGH (data corruption under concurrent invocation; resolved, record only)
- Group: correctness/concurrency — Rank: 1/5 (critical pattern, fixed)
- Area: correctness — concurrency / single-writer assumption
- Rank rationale: silent media + state corruption from a plausible operator action
  (two `voyage run`, or `run` + manual `commit`); pre-fix no lock existed anywhere.

## Technical description

Pre-fix, segment creation and state advance were lock-free:

```python
# voyage/supervisor.py:1023,1225-1235 (pass 1)
segment.mkdir(parents=True, exist_ok=True)   # no exclusive create
...
fresh = read_state(self._run_dir)            # re-read, then blind advance
fresh.next_segment_number = number + 1       # number is stale local
fresh.committed_segments += 1
fresh.timeline_frames += frames
```

`rg lock voyage` returned only `_call_lock` (per-process threads). Two processes
read `N`, both `mkdir(exist_ok)` the same `segments/N/`, both `ffmpeg -y` to the
same `video.mp4`/`audio.wav`, both compute DONE, then both read-modify-write
`state.json`. Outcomes: interleaved/truncated media, `committed_segments`
double-counted for one DONE dir (`validate` then reports `state claims 2, found 1
DONE`), or `timeline_frames` double-added. Pause/stop races were the same class.

Live state (re-verified 2026-09-30): `_held_run_lock`
(`voyage/supervisor.py:352-368`) — `fcntl.flock(LOCK_EX | LOCK_NB)` on
`<run>/state.json.lock`, holder pid recorded, second writer fails fast with
`FatalWorkerError`; commit runs under it (`voyage/supervisor.py:1846`).

## Why this is an issue

- The failure mode is corruption, not a clean error — worst kind.
- `validate`'s contiguous-numbering/orphan checks assume single-writer history;
  multi-writer breaks the invariant they verify.
- Crash-matrix tests only cover single-writer crashes (correctly), leaving
  concurrent invocation entirely untested pre-fix.

## Evidence

Live verification 2026-09-30:

```
$ rg -n "flock|O_EXCL|state.json.lock|lockfile|fcntl|run_lock" voyage/supervisor.py
16: import fcntl
345:     def _held_run_lock(self) -> Iterator[None]:
348:         `fcntl.flock(LOCK_EX | LOCK_NB)` on `<run>/state.json.lock`: the
353:         lock_path = self._run_dir / "state.json.lock"
358:                 fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
371:                 fcntl.flock(lock_fd, fcntl.LOCK_UN)
1838:         with self._held_run_lock():
```

Pass-1 evidence: `rg -n "flock|O_EXCL|state.json.lock|lockfile" voyage/ tests/` →
NO HITS; only `SubprocessWorker._call_lock = threading.Lock()` (thread-local).

## Reproduction

1. `Supervisor(run_dir, config).run_segments(10)` in two parallel shells on one
   run dir.
2. Pre-fix observed: duplicate `00000N` overwrites and
   `committed_segments != DONE count`. Post-fix: the second supervisor exits fast
   with `FatalWorkerError: run <dir> is locked by pid X`.

## Source references

- `voyage/supervisor.py:344-381` (lock + holder), `:1838-1842` (commit under lock);
  `voyage/rpc.py:127-131` (`_call_lock` scope — threads only, unchanged).

## Resolution candidates

1. (Landed) Single-writer lock: `fcntl.flock(run_dir/state.json.lock, LOCK_EX)`
   held for the whole commit (with `O_CREAT`, pid record, non-blocking fail-fast).
2. Document `run` as singleton in CLI help + DESIGN failure-policy docs.
3. Second process exits with a clear "run locked by pid X" `FatalWorkerError`, not
   queueing behind indefinitely (explicit `--wait` would be a separate feature).

## Online references

- Python `fcntl.flock` — "Perform the lock operation op on file descriptor fd"
  (`LOCK_EX | LOCK_NB` for non-blocking exclusive acquisition; locks die with the
  process / on close):
  https://docs.python.org/3/library/fcntl.html
- Race conditions from check-then-act on shared files (CWE-362):
  https://cwe.mitre.org/data/definitions/362.html

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Resolution batch 3: `fcntl` run lock + fail-fast second writer landed.
- 2026-09-30: re-verified live (lock, holder, commit coverage all present);
  reconstructed from archived pass-1 text (commit `b5d7dda`). Status → resolved.
