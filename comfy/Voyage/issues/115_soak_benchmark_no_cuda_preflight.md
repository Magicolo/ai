# 115 — `soak` and `benchmark video/audio` skip the CUDA fast-fail: torch-less runs die late at worker init instead of preflight

- **Severity:** LOW (error quality + wasted setup: the condition is knowable in 3 lines, the failure arrives minutes later inside a worker)
- **Track:** second-pass TUI/CLI edges (preflight coverage across verbs)
- **Verified:** 2026-09-30 by source read + probe (lines as-read; concurrent edits in `cli.py` do not touch these paths)

## File:line (live-verified)

- `voyage/cli.py:1207` (`_require_cuda_stack`: fast-fail when a CUDA backend is configured but torch is not importable)
- `voyage/cli.py:520` (`cmd_run`: gates on it), `:1270-1272` (`cmd_generate`: gates on it + `_warn_if_no_cuda`)
- `voyage/cli.py:1458` (`cmd_benchmark`: `video`/`audio` targets call `supervisor.start_workers()` with no gate)
- `voyage/cli.py:1546` (`cmd_soak`: prints the rule line, then `Supervisor(...).run_segments` with no gate)
- Distinct from 021: that issue files set *membership* (polluted `acestep`, missing `mmaudio`, offenders ignoring `sfx.backend`). This issue is call-site *coverage*: two worker-spawning verbs never consult the set at all, whatever it contains.

## Description

`run` and `generate` fail fast with the actionable CUDA message (image + `--gpus all` pointer) before starting anything. `benchmark video|audio` and `soak` spawn the same workers with no check: on a torch-less image (slim `voyage:latest`, or a direct `docker run` without the run.sh auto-selection) the run proceeds to worker init and dies there — for soak, after printing the rule header, i.e. looking healthy until the first segment's worker crashes. The end-to-end benchmark target is unaffected (hardcoded `backend="fake"`, `:1462`), which makes the gap easy to miss in review: the verb is half-guarded by construction.

## Rationale

- Preflight exists precisely because worker-init failures are noisy and late (021's rationale: "late failure inside the worker wastes the whole render"). Soak is the longest-running verb; discovering a missing GPU stack inside it wastes the most wall time.
- The fix is the same two lines `run` already has. The risk of filing this separately from 021 is duplication; the reason it is separate is that fixing 021's set membership does not fix these call sites, and fixing these call sites does not fix 021's set — each fix leaves the other bug live.

## Evidence (source + probe, 2026-09-30)

```
$ grep -n "_require_cuda_stack\|_warn_if_no_cuda" voyage/cli.py
481:    if not _require_cuda_stack(config):     # cmd_run
1168:def _require_cuda_stack(...)               # def
1187:def _warn_if_no_cuda(...)                   # def
1230:    if not _require_cuda_stack(config):     # cmd_generate
1232:    _warn_if_no_cuda(config)                # cmd_generate only
# cmd_benchmark (:1416), cmd_soak (:1504): zero hits
```

`cmd_soak` loads the run config (`:1517`) — the object the gate needs is already in hand; the gate call is purely missing, not blocked on refactoring.

## Repro

```bash
# Slim image (no torch), any CUDA-backend run dir R:
./Voyage/scripts/run.sh benchmark video --run R   # dies at worker init, not at preflight
./Voyage/scripts/run.sh soak --run R --segments 1 # prints "voyage soak · 1 segments", then worker init fails
# Compare: ./Voyage/scripts/run.sh run --run R --segments 1  # fast-fails with the CUDA message
```

## Fix candidates

- Add the `cmd_run` two-liner (`if not _require_cuda_stack(config): return 1`) to `cmd_benchmark` (video/audio targets, after `_load_run`) and `cmd_soak` (after `_load_run`, before the rule line so the header never prints for a doomed run).
- Optionally `_warn_if_no_cuda` in soak (CUDA device + no visible GPU fails at worker init even when torch exists).
- Test: torch-unavailable + CUDA-backend config → `benchmark video`/`soak` return 1 with the CUDA message and never start workers (monkeypatch `_torch_available`, as `test_cuda_fast_fail_returns_to_form_with_error` does for the TUI path).

## Refs

- Issue 021 (set membership — the other half); `tests/test_tui_app.py:863-897` (fast-fail test pattern to mirror); `Voyage/scripts/run.sh` (auto-selects the video image for CUDA backends — the mitigation that makes this low rather than medium).
