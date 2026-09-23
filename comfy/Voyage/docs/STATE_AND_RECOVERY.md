# STATE_AND_RECOVERY — persistence invariants and crash scenarios

## Invariants (all enforced by `voyage validate`, testable via `validate_run`)

1. **DONE gates commitment.** Only `segments/NNNNNN/` dirs containing a
   `DONE` marker count as committed. Everything else is scratch and is
   ignored by resume, finalize, and scoreboard.
2. **Contiguous numbering from 000000.** No gaps, no renames; the next
   commit always reuses `state.next_segment_number`.
3. **State follows artifacts.** `state.json` advances only after media,
   metadata, checksums, and `DONE` are durable. `committed_segments`
   equals the DONE count; `timeline_frames` equals the frame sum.
4. **Checksums are recomputed, not trusted.** `sha256.json` is rewritten
   at commit and re-hashed by validate and finalize (§56 step 4).
5. **Atomic state writes.** All JSON state goes temp + fsync + rename,
   plus directory fsync after `DONE`. Concept vectors/index appends and
   the audio ledger are fsynced; torn files are impossible, old files
   survive a crash.
6. **Frame-range sanity + A/V alignment.** Finalize checks monotonic
   frame ranges and ≤0.6 s audio/video drift per segment (`--skip-bad`
   skips corrupt segments with a warning instead of aborting).
7. **No orphan partials.** Any `*.partial` anywhere under `segments/`
   fails validation — a leftover means an interrupted commit to inspect.
8. **Tapes never cross numerics.** `recovery.pt` records its `fp8|bf16`
   profile; resume refuses cross-numeric tapes.

## Crash scenarios

| Crash point | State left behind | Recovery |
|-------------|-------------------|----------|
| Worker SIGKILL mid-op | broken pipe → `RecoverableWorkerError` | restart (budget 3/run, then circuit-breaker → FAILED), resume from tape |
| Supervisor SIGKILL before DONE | partial dir, no DONE | ignored; next run reuses the number, overwrites media |
| Between DONE rename and state advance | DONE without count | next run re-commits the same number cleanly |
| Mid-`state.json` write | old file intact (atomic replace) | continue from old state |
| Disk-full mid-commit | `DiskSpaceError` → `PAUSED_DISK_FULL` | free space, `run` again; precheck re-pauses if still full |
| Repeated worker failure | `circuit_breaker_open` metric, FAILED | fix cause (models/VRAM), `run` resumes from last commit |
| Hung worker (no output) | RPC select-deadline expires (default 600 s) | treated as recoverable → restart path |
| Supervisor SIGKILL mid-finalize | sources untouched (finalize never mutates) | re-run finalize |

Key knobs: `[voyage] max_worker_restarts = 3`, `rpc_timeout_seconds = 600`,
`min_free_space_gib` (20.0 spec default, 5.0 in dev TOML — raise for
production). Crash-injection hook for tests: `Supervisor.inject_worker_crash(name)`.
Full matrix pinned in `tests/test_crash_matrix.py` + `test_failure_policy.py`.
