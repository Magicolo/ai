# 019 — `run_capture` has no `timeout=`; wedged ffmpeg/ffprobe wedges the commit forever

- Severity: MEDIUM
- Group: correctness/liveness — Rank: 2/5 (liveness-adjacent; no outer watchdog)
- File:line: `voyage/media.py:75-76` (`run_capture`)

## Technical description

Live code (re-verified 2026-09-30):

```python
# voyage/media.py:75-76
def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)
```

No `timeout=`. Every ffmpeg/ffprobe invocation in the tree funnels through this helper (`probe`, `slice_take`, `assemble_segment_audio`, `_blend_pair`, finalize concat/encode, `sample_frames`, VAE-adjacent probes). A wedged child (GPU-ffmpeg hang, NFS-stalled input, driver-level wait, `minterpolate` pathological frame) blocks the calling thread indefinitely:

- The commit path holds the run lock (`_held_run_lock`, `supervisor.py:344-381`) while blocked → `voyage pause`/`stop`/`status` from another process pile up on the lock; `STOP_REQUESTED` (written lock-free by `cli._set_status`) is never observed because the commit never reaches the next boundary check.
- The RPC restart budget never engages: the hang is *below* the RPC layer (local subprocess, not a worker call), so no `RecoverableWorkerError`, no `worker_restart` metric, no circuit breaker.
- `run_segments` has no outer watchdog, so status rests at RUNNING forever.

Contrast the RPC layer, which *does* bound every worker call (`rpc.py:DEFAULT_RPC_TIMEOUT_SECONDS = 600.0`, non-blocking reader with deadline, issue 001 fixed). The local-subprocess layer has no equivalent.

## Why it matters

- Liveness: one stuck ffmpeg (observed in the wild with `minterpolate` + `concat` on corrupt segments) wedges the autonomous voyage with zero diagnostics — worse than a crash, because monitoring sees RUNNING, not FAILED.
- Lock interaction (080 family): the wedged commit holds `state.json.lock`, turning a media stall into a control-plane stall.
- The fix is mechanical (`timeout=` + `TimeoutExpired → MediaError`), and the taxonomy already exists.

## Live evidence

```
$ grep -n "def run_capture" -A2 comfy/Voyage/voyage/media.py
75:def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
76:    return subprocess.run(argv, capture_output=True, text=True, check=False)
```

- Zero `timeout` occurrences in `voyage/media.py` (verified via `grep -c "timeout" voyage/media.py` → only comments, no `subprocess` timeout).
- Callers all share the helper: `grep -n "run_capture(" voyage/media.py` lists `probe`, `slice_take`, `assemble_segment_audio`, `_blend_pair`, finalize paths — every one inherits the unbounded wait.

Host cannot run ffmpeg reliably here, but the stdlib contract is sufficient: `subprocess.run(..., timeout=None)` (default) waits indefinitely — documented: "If the timeout expires, the child process will be killed and waited for. ... TimeoutExpired will be raised."

## Repro steps

1. Replace `ffmpeg` on PATH with `#!/bin/sh\nsleep 3600` (or point a take at an NFS-frozen file).
2. Start a fake-backend commit that must slice/assemble audio.
3. Observed: `commit_one_segment` blocks in `run_capture` past any `rpc_timeout_seconds`; `Ctrl-C`/SIGINT is the only exit; no `worker_restart` metric; lock held.
4. Expected: `TimeoutExpired` mapped to `MediaError` within a bounded budget, commit fails safe, restart budget / `--skip-bad` engages.

## Fix candidates

1. (Preferred) Add a module constant (e.g. `FFMPEG_TIMEOUT_SECONDS = 600.0`, mirroring `DEFAULT_RPC_TIMEOUT_SECONDS`) and pass `timeout=` in `run_capture`; catch `subprocess.TimeoutExpired` and raise `MediaError(f"... timed out after {N}s: {argv[0]} ...")`. One-line change fixes all callers; per-op overrides (probe vs finalize encode) can follow.
2. Kill-group hygiene: `subprocess.run` with `timeout` kills the child but not grandchildren; for ffmpeg filter graphs that spawn workers, consider `start_new_session=True` + killpg on timeout (follow-up, not required for the first fix).
3. Regression test: stub `subprocess.run` to sleep past a tiny timeout (monkeypatch the constant); assert `MediaError`, not a hang. Liveness test: `probe` with a wedged ffprobe returns within ~2× timeout.

## References

- `voyage/media.py:75-76` (helper), callers at `:80` (`probe`), `:175` (`slice_take`), `:253+` (assembly), `:1050+` (finalize).
- Contrast `voyage/rpc.py:59-63,192-254` (bounded worker calls, issue 001) and `voyage/supervisor.py:110-120` (`GAUGE_TIMEOUT_SECONDS`/`EMBED_TIMEOUT_SECONDS` — bounded *worker* probes, unbounded *local*ffmpeg).
- Python `subprocess.run(timeout=...)`: https://docs.python.org/3/library/subprocess.html#subprocess.run

## Progress log

- 2026-09-30: re-verified live in-container before touching anything — holds as-read: `voyage/media.py:98-99` (`subprocess.run(argv, capture_output=True, text=True, check=False)`, zero `timeout` occurrences in the file). Every ffmpeg/ffprobe call site (`probe`, `slice_take`, `assemble_segment_audio`, `_blend_pair`, finalize concat/encode) funnels through the helper and inherits the unbounded wait.
- 2026-09-30 (TDD red): wrote `Voyage/tests/test_media_robustness_rank2.py` first; the three 019 tests failed as required (`FFMPEG_TIMEOUT_SECONDS` missing, no `timeout` param, `TimeoutExpired` escaping raw).
- 2026-09-30 (implement, `voyage/media.py` only — candidate 1): added `FFMPEG_TIMEOUT_SECONDS = 600.0` (mirrors `rpc.DEFAULT_RPC_TIMEOUT_SECONDS` / `[voyage] rpc_timeout_seconds`); `run_capture(argv, timeout=FFMPEG_TIMEOUT_SECONDS)` passes `timeout=` through and maps `subprocess.TimeoutExpired → MediaError("<bin> timed out after <N>s")`. Existing call sites unchanged (default keeps them bounded with no config round-trip); callers may thread the configured value explicitly later.
- 2026-09-30 (TDD green + gates): 18 passed in the new file; related suites 93 passed + 1 torch-gated skip; `ruff check` + `ruff format --check` + `mypy` (strict) clean on all touched files.

## Resolution

- Verdict: fixed in scope (helper + constant). All callers inherit the bound at once; the commit path can now fail safe (restart budget / `--skip-bad`) instead of wedging the run lock at RUNNING.
- Files changed: `Voyage/voyage/media.py` (`FFMPEG_TIMEOUT_SECONDS` + `run_capture`), `Voyage/tests/test_media_robustness_rank2.py` (new: default-value pin, `TimeoutExpired→MediaError`, kwargs-threading pin).
- Test evidence: `test_run_capture_exposes_default_timeout` (constant 600.0 + `timeout` param present), `test_run_capture_timeout_expired_maps_to_media_error` (stubbed `subprocess.run` raising `TimeoutExpired` → `MediaError` matching "timed out"), `test_run_capture_threads_timeout_to_subprocess` (timeout=7.5 reaches `subprocess.run` kwargs).
- DESIGN.md as-built proposal (not applied — DESIGN.md untouched per directive): in the media/RPC section, add "local ffmpeg/ffprobe spawns carry the same 600 s budget as worker RPC (`FFMPEG_TIMEOUT_SECONDS`, threadable from `[voyage] rpc_timeout_seconds`); expiry is a `MediaError`, never a wedged commit."
- Residuals / follow-ups: per-op overrides (short probe vs long finalize encode) not added — one budget for all callers for now; kill-group hygiene (`start_new_session` + `killpg` for ffmpeg filter-graph grandchildren, candidate 2) left as follow-up.
