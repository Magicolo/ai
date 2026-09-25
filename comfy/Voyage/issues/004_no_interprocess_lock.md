# 004 — No inter-process mutual exclusion: two supervisors corrupt the same segment + state

- Status: resolved 2026-09-25 (supervisor track; pause-race + CLI-doc follow-ups logged)
- Severity: critical (data corruption under concurrent invocation)
- Area: correctness — concurrency / single-writer assumption
- Rank rationale: silent media + state corruption from a plausible operator action
  (two `voyage run`, or `run` + manual `commit`); no lock exists anywhere.

## Technical description

```python
# voyage/supervisor.py:1023,1225-1235
segment.mkdir(parents=True, exist_ok=True)   # no exclusive create
...
fresh = read_state(self._run_dir)            # re-read, then blind advance
fresh.next_segment_number = number + 1       # number is stale local
fresh.committed_segments += 1
fresh.timeline_frames += frames
```

`rg lock voyage` returns only `_call_lock` (per-process threads). Two processes
read `N`, both `mkdir(exist_ok)` the same `segments/N/`, both `ffmpeg -y` to the
same `video.mp4`/`audio.wav`, both compute DONE, then both read-modify-write
`state.json`. Outcomes: interleaved/truncated media, `committed_segments`
double-counted for one DONE dir (`validate` then reports `state claims 2, found 1
DONE`), or `timeline_frames` double-added. Pause/stop races (`_pause_requested` +
commit-tail `write_state`) are the same class.

## Why this is an issue

- The failure mode is corruption, not a clean error — worst kind.
- `validate`'s contiguous-numbering/orphan checks assume single-writer history;
  multi-writer breaks the invariant they verify.
- Crash-matrix tests only cover single-writer crashes (correctly), leaving this
  entirely untested.

## Evidence

- `rg -n "lock|flock" Voyage/voyage` → only `SubprocessWorker._call_lock`
  (thread-local). No `fcntl`, no lockfile, no `O_EXCL` segment creation.

Re-verified 2026-09-25 (live tree):

```
$ rg -n "flock|O_EXCL|state.json.lock|lockfile" voyage/ tests/ -g '*.py'
NO HITS (no inter-process lock)
$ rg -n "_call_lock" voyage/rpc.py
94:        self._call_lock = threading.Lock()
156:        with self._call_lock:
```

## Reproduction

1. `Supervisor(run_dir, config).run_segments(10)` in two parallel shells on one
   run dir.
2. Observe duplicate `00000N` overwrites and `committed_segments != DONE count`.

## Source references

- `voyage/supervisor.py:1023`, `:1225-1235`; `voyage/rpc.py:90-94` (lock scope).

## Resolution candidates

1. Single-writer lock: `fcntl.flock(run_dir/state.json.lock, LOCK_EX)` held for the
   whole `commit_one_segment` (or at minimum around number-allocation +
   state-advance, with `O_CREAT|O_EXCL` segment-dir creation failing when taken).
2. Document `run` as singleton in CLI help + DESIGN §69-adjacent failure-policy docs.
3. Second process should exit with a clear "run locked by pid X" `FatalWorkerError`,
   not queue behind indefinitely (or support `--wait` explicitly).

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Open: implement lock + test (two supervisors, second fails fast).
- 2026-09-25 (repair pass): refs verified current (`supervisor.py:1023,1225`);
  Evidence enriched with live `rg` output (no flock/lockfile);
  `## Why this is an issue` already present, no change.
- 2026-09-25 (RESOLVED, supervisor track): implemented candidate 1.
  `Supervisor._held_run_lock` (`voyage/supervisor.py:228-257`) holds
  `fcntl.flock(LOCK_EX | LOCK_NB)` on `<run>/state.json.lock` (holder pid
  recorded in the file) for the whole commit: `commit_one_segment` is now a
  thin wrapper acquiring the lock around `_commit_one_segment_locked`
  (`voyage/supervisor.py:1264-1272`). The second writer fails fast with
  `FatalWorkerError` naming the holder pid — never queues, never
  interleaves. The lock dies with the process, so no stale-lock recovery
  exists by design. Regression test
  (`test_second_writer_fails_fast_when_locked` in
  `Voyage/tests/test_commit_hardening.py`) holds the lock on one
  supervisor and asserts the second fails in <5 s. OBSERVED FLAKE (same
  class, out of scope): `test_pause_mid_run_stops_at_boundary` failed once
  under full-suite load (10 commits landed, pause missed) and passes in
  isolation — the commit-tail `write_state(fresh)` can clobber an
  externally-written PAUSE_REQUESTED between its read and write (the race
  this issue's description already names). Candidate: hold the same run
  lock around pause/stop-flag transitions, or make the tail advance
  preserve an externally-set terminal status. CLI singleton doc (candidate
  2) needs `cli.py` (out of scope) — deferred. Gates: full
  `Voyage/scripts/gates.sh` green (626 passed).
