# 099 — Control-plane lost update: fcntl run lock excludes second supervisors but not status writers

**Severity:** MEDIUM

**File:line (verified live 2026-09-30; `git status` shows uncommitted concurrent edits in `voyage/cli.py`, `voyage/tui_state.py`, `Voyage/tests/test_generate.py` — `cli.py` lines below are as-read, not HEAD):**
- `voyage/supervisor.py:352-390` (`_held_run_lock` — fcntl lock held for one commit)
- `voyage/supervisor.py:1789-1799` (`_commit_segment` step 6: `fresh = read_state(...)` → mutate counters → `write_state(fresh)`, never touches `status`)
- `voyage/supervisor.py:1846-1850` (`commit_one_segment` holds the lock across the whole commit body)
- `voyage/supervisor.py:608-619` (`_pause_requested` / `_stop_requested` re-read `state.json` at every segment boundary)
- `voyage/cli.py:750-770` (`_set_status`: `read_state` → assign → `write_state`, **no lock**; `cmd_pause` / `cmd_resume` / `cmd_stop` all via `_set_status`)
- `voyage/tui.py:1059-1071` (Stop button: `read_state` → `status = "STOP_REQUESTED"` → `write_state`, **no lock**)
- `voyage/persistence.py:82-99` (`write_state` is atomic on disk but lock-free)

**Description:**
The run lock (`fcntl.flock(LOCK_EX | LOCK_NB)` on `<run>/state.json.lock`, `supervisor.py:355-358`) is held for the entire commit (`commit_one_segment`, lines 1838-1839), and the docstring (line 346) says "held for one commit". But the only writers that honor it are supervisors: the CLI control plane (`voyage pause/resume/stop` via `_set_status`, `cli.py:711-720`) and the TUI Stop button (`tui.py:1058-1069`) perform an unlocked read-modify-write of the same `state.json`. Meanwhile the commit's own state advance is also a read-modify-write that preserves whatever `status` it read:

```python
# supervisor.py:1789-1799
fresh = read_state(self._run_dir)
fresh.next_segment_number = number + 1
fresh.committed_segments += 1
...
fresh.last_error = None
write_state(self._run_dir, fresh)
```

`fresh.status` is never assigned here, so the commit writes back the status value sampled at line 1781. Interleaving:

1. Commit reaches step 6, reads `fresh` (`status == "RUNNING"`).
2. Operator runs `voyage stop` → `_set_status` writes `status == "STOP_REQUESTED"`.
3. Commit finishes metadata/checksums/DONE (including two `sha256_file` passes over full media — a seconds-long window on large segments) and executes line 1799, restoring `status == "RUNNING"`.

The stop is clobbered. The run loop only consults the file at segment boundaries (`_stop_requested`, lines 617-619; `run_segments` docstring, lines 623-629: "each loop iteration re-reads state.json"), so the run sails past the operator's stop and keeps committing. The reverse direction exists too: `_pause_requested` (lines 608-615) and the loop's resting-state writes (lines 701-711) are unlocked read-modify-writes that can write back stale counters over a concurrent commit's advance (losing `committed_segments += 1` / `timeline_frames += frames`).

**Rationale:**
The lock's documented purpose (lines 353-358: "the second supervisor fails fast … instead of interleaving media + state writes") is defeated for the highest-stakes writer — the human asking the run to stop. A stop that silently does nothing is worse than no stop command at all: the operator believes the run is stopping while GPU work continues, and `stop --finalize` then finalizes a run that kept moving. The failure is timing-dependent (needs the operator write to land inside the step-6 window), so it presents as a flaky "stop didn't take" — exactly the class of bug that survives manual testing.

**Live evidence (current tree):**
```
$ git status --short
 M Voyage/tests/test_generate.py
 M Voyage/voyage/cli.py
 M Voyage/voyage/tui_state.py
```
`cli.py:750-760` (as-read live):
```python
def _set_status(run: Path, status: str) -> int:
    try:
        state = read_state(run)
    except StateError as exc:
        ...
    state.status = status  # type: ignore[assignment]
    write_state(run, state)
```
No `fcntl`, no `_held_run_lock`, no CAS — plain last-writer-wins. Same shape at `tui.py:1059-1071` and `supervisor.py:608-615` / `701-711`.

Stdlib analog of the exact primitive (no voyage imports — pydantic is container-only):
```
$ python3 -c "<read-modify-write interleave demo>"
after interleaved commit: {'n': 1, 'status': 'RUNNING'} <- STOP_REQUESTED clobbered = lost update
```
The commit's counter advance (`n: 0 → 1`) is correct while the operator's status write is destroyed — precisely the lines 1789-1799 shape.

**Repro (CPU, no GPU):**
1. `init` a run with the `fake` video backend; `start_workers()` + one `commit_one_segment()` in process A (holds `_held_run_lock` for the whole body).
2. While A is inside step 6 (between the `fresh = read_state` at line 1789 and `write_state` at 1799 — widen the window by pointing the run at a large pre-existing segment so `sha256_file` takes seconds), run `voyage stop --run <dir>` in process B.
3. Read `state.json`: `status` is back to `RUNNING`; a subsequent `run_segments` iteration does not stop. Reverse repro: pause B between A's step-6 read and write, then diff `committed_segments` against the segment dirs on disk.

**Fix candidates:**
- Take `_held_run_lock` (or a shared `fcntl` helper in `persistence.py`) in `_set_status` (`cli.py:750`) and the TUI Stop handler (`tui.py:1059-1071`): lock acquisition is `LOCK_NB` + fail-fast today, so add a blocking variant for control-plane writers — a stop must wait out an in-flight commit, never fail-loud against it.
- Alternatively compare-and-swap in `_commit_segment`: re-read `status` immediately before line 1799 and preserve any `*_REQUESTED` value over the stale copy (still leaves the reverse direction — loop resting writes clobbering counters — so the lock is the complete fix; CAS is the minimal one).
- Same treatment for `_pause_requested` (lines 608-615) and the resting writes (lines 701-711): they are read-modify-write on shared state and belong inside the lock or behind CAS.
- Test: two-process (or two-thread, same primitives) interleaving test asserting a `STOP_REQUESTED` written during step 6 survives the commit, and a commit's counters survive a concurrent pause.

**Refs:**
- Overlaps with 097 (the lock hardened there is the lock the control plane bypasses here) — ownership stays here (control-plane writers).
- In-tree: `voyage/supervisor.py:352-390` (lock), `:1789-1799` (state advance), `:623-629` ("each loop iteration re-reads state.json so `voyage pause` / `voyage stop` … takes effect at the next segment boundary" — the contract this bug breaks); `voyage/cli.py:750-770`; `voyage/tui.py:1059-1071`; `voyage/persistence.py:82-99`.
- `fcntl.flock` semantics (Python docs, https://docs.python.org/3/library/fcntl.html): locks are advisory — "only … processes … that also use flock" are excluded. Quoting the mechanism: a lock that one side never acquires excludes nobody on that side. Verified by code inspection (no `fcntl` import or lock call anywhere in `cli.py`/`tui.py`).
