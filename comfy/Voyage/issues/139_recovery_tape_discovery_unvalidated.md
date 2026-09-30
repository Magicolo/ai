# 139 — Recovery-tape discovery returns the newest `recovery.pt` with zero validation

- Severity: MEDIUM
- Area: supervisor resume path — trust boundary
- Files (as-read 2026-09-30):
  - `voyage/supervisor.py:530-563` (`_latest_recovery_tape` + `_resume_video_worker`)
  - Consumer: `voyage/supervisor.py:559-562` (`{"recovery_path": str(tape)}` → video `resume` op)

## Technical description

`_latest_recovery_tape` (`voyage/supervisor.py:530-544`, as-read) walks the segments dir in
reverse-sorted name order, returns the first `recovery.pt` under a `DONE`-marked segment, and
`_resume_video_worker` (`:546-562`) hands it straight to the freshly restarted video worker:

```python
# voyage/supervisor.py:530-562 (as-read, abridged)
def _latest_recovery_tape(self) -> Path | None:
    segments_root = self._run_dir / paths.SEGMENTS_DIRNAME
    if not segments_root.exists():
        return None
    for segment in sorted((p for p in segments_root.iterdir() if p.is_dir()),
                          key=lambda p: p.name, reverse=True):
        if (segment / paths.DONE_MARKER).exists():
            tape = segment / "recovery.pt"
            if tape.exists():
                return tape
    return None

def _resume_video_worker(self, segment_id: str) -> None:
    tape = self._latest_recovery_tape()
    if tape is None:
        return
    result = self._call_with_restart(
        self._video, "video", segment_id, "resume", {"recovery_path": str(tape)})
```

No validation at discovery: no `is_file` (a directory named `recovery.pt` passes `.exists()`),
no size bounds, no readability check, no run-relative confinement (`str(tape)` is an absolute
path built from `self._run_dir`, but a symlinked segment dir or a symlinked `recovery.pt`
pointing outside the run is followed silently), no schema/shape check before the worker
`torch.load`s it. A torn write (crash between `torch.save` and `DONE`), a truncated file, or a
planted symlink all surface one layer later — inside the worker's `resume` handler or as a
confusing deserialization traceback — instead of at discovery where the supervisor could skip
to the next-newest tape or fall back to a fresh stream.

First-segment behavior (`None` → fresh stream) is correct and stays.

## Why this is an issue

- Resume is on the failure path already (post-restart): a second failure here burns the shared
  restart budget (`resume` retries through `_call_with_restart`) on an input that was knowably
  bad before any RPC.
- `reverse=True` name-sort assumes zero-padded contiguous ids (`%06d`); a stray non-segment dir
  sorts arbitrarily and is only filtered by `is_dir` + `DONE` presence — thin filtering for the
  input that re-seeds the whole causal stream.
- Not a duplicate of 006 (worker-reported counts/tapes trusted blindly at commit), 016 (tape
  symlink/TOCTOU at the checker), or 123 (`resume trusts tape blindly` at the worker): this is
  the supervisor-side *discovery* step between them — picking *which* tape to trust.

## Live evidence

Live re-verification 2026-09-30 (read-only; probes per task brief ran in `voyage:latest` CPU-only).
Track A draft command+output bundle (`ses_f0fbea412ffeh3V7K1SlQpXjsv`) was not recoverable from
this writer's context, so evidence below is the as-read code, not invented command output:

```
$ sed -n '530,563p' voyage/supervisor.py
  530:    def _latest_recovery_tape(self) -> Path | None:
  531:        """Newest committed segment's recovery.pt, or None (DESIGN §27)."""
  ...
  540:            if (segment / paths.DONE_MARKER).exists():
  541:                tape = segment / "recovery.pt"
  542:                if tape.exists():
  543:                    return tape
  556:        tape = self._latest_recovery_tape()
  560:        result = self._call_with_restart(... "resume", {"recovery_path": str(tape)})
```

No `is_file`, no size/read check, no symlink confinement, no fallback to older tapes in this
function's body (as-read).

## Minimal repro

1. Commit segment `000000` (produces `segments/000000/recovery.pt` + `DONE`).
2. Truncate the tape: `: > segments/000000/recovery.pt` (0 bytes), or replace it with a symlink
   to `/etc/hostname`.
3. Kill the video worker mid-next-segment so `_resume_video_worker` runs.
4. Observed: `resume` RPC fails inside the worker (deserialization/symlink read) and consumes
   restart budget; expected: discovery skips the bad tape (or falls back to fresh stream) with a
   `video_resume_skipped` metric.

## Fix candidates

1. (Preferred) Validate at discovery: `is_file` + `not is_symlink` (or resolve-and-confine via
   the issue-016 checker), non-zero size with an upper bound, then skip-and-continue to the next
   newest `DONE` segment on failure instead of returning the bad tape. Emit a metric per skip.
2. Reuse the tape checker both sides: discovery resolves through the same run-relative
   confinement the worker enforces, so supervisor and worker agree on what "valid tape" means.
3. Add a `recovery.pt` entry to `sha256.json`/segment checksums so torn writes fail loud at
   `validate_run` time too (pairs with the discovery-time skip).
4. Regression tests: 0-byte tape → falls back to older tape/fresh stream; symlink tape →
   skipped; directory named `recovery.pt` → skipped; missing segments dir → `None`.

## References

- In-tree: `voyage/supervisor.py:530-563`; `voyage/paths.py` (`SEGMENTS_DIRNAME`, `DONE_MARKER`,
  `format_segment_id`); `DESIGN §27/§27.1` (recovery tapes + resume hook).
- Neighbor issues: 006 (untrusted worker reports), 013 (DONE-before-state window that can leave
  the torn tape this discovers), 016 (tape symlink/TOCTOU), 122 (tape write without fsync),
  123 (worker-side blind resume trust).
- External:
  - https://docs.python.org/3/library/pathlib.html#pathlib.Path.exists (`exists()` follows symlinks;
    `is_file()` + `is_symlink()` distinction)
  - https://pytorch.org/docs/stable/generated/torch.save.html (torn-save hazard on crash)

## Investigation log

- 2026-09-30: filed by Track A sweep; live re-verified via Read (concurrent uncommitted edits
  noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`,
  `config/persistence/rpc/supervisor` — citations are as-read values above).

## Progress log

- 2026-09-30 (surface-rank2 track): premise PARTIALLY REFUTED on live re-read — the filed "zero validation" no longer holds: `_latest_recovery_tape` (`voyage/supervisor.py:714-755`) already carries the issue-016 discovery mirror (resolve-containment + `is_file` on the resolved tape + loud `recovery_tape_skipped` metric + skip-and-continue), covered by `test_latest_recovery_tape_skips_planted_entries` / `..._skips_directory_tape`. Remaining gap found live: a 0-byte torn tape (crash between torch.save and DONE) passes every existing check and is handed to the worker, burning restart budget. Size bounds are otherwise 171's scope (upper bound open there).
- Fix (discovery region only): skip zero-byte/un-stat-able tapes with a loud `recovery_tape_skipped` (reason `empty file (torn write)`) and fall through to the next-newest DONE segment (fresh stream when none remain). No change to `_resume_video_worker`, commit, prefetch, or gauge regions.
- Evidence: 2 new 139 tests failed pre-fix, pass post-fix; supervisor-hardening + phase2 suites green; ruff + format-check + mypy strict clean.

## Resolution

- FIXED (residual leg) 2026-09-30: discovery now skips is_file + containment + empty tapes with metrics; the filed zero-validation premise itself had already landed via the 016 mirror. Upper size bound stays with 171.
