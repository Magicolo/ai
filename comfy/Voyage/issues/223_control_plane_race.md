# 223 — Control-plane `read-then-write` race survives `_write_state_preserving_control_plane` (MEDIUM)

## Technical description
`supervisor.py:2386-2403` re-reads `state.json` to honor
`STOP/PAUSE_REQUESTED` written lock-free by `voyage stop/pause`, then
`write_state`. A request landing **after** the live read but **before**
`write_state` is still clobbered. The window is microseconds (one
`read_state` + one atomic write), but the commit path holds it on *every*
segment, and `test_recovery.py:67-89` already documents the sibling race
("pre-existing race … silently clobbered").

```python
try: live_status = read_state(...).status / except VoyageError: live_status = fresh.status
if live_status in (...): fresh.status = live_status
write_state(self._run_dir, fresh)
```
No lock, no `O_EXCL`/CAS — two writers (`commit` under `_held_run_lock`,
`stop/pause` without it) interleave by design.

## Rationale
`stop`/`pause` is the only operator brake on an infinite run. Losing it
means the run sails past the boundary it was told to rest at; the next
boundary re-reads, so the delay is one segment — but on 4-minute LTX
segments that is 4 minutes of unwanted GPU burn.

## Live evidence
Code shape (race is structural, not probabilistic in unit tests) — source
printed via `inspect.getsource`, see above.

## Repro
Thread A loops `_write_state_preserving_control_plane`, thread B writes
`STOP_REQUESTED` between A's `read_state` return and `write_state` entry
(inject via monkeypatched `read_state` barrier); assert B's request is lost.

## Source refs
- `Voyage/voyage/supervisor.py:2386-2403,2612-2624,2707-2793`
- `Voyage/tests/test_recovery.py:73-80` comment

## Online sources
- https://pkg.go.dev/github.com/larsartmann/go-atomic-write%40v0.3.0 —
  fingerprint + lock + verify (read-verify-write, not read-then-write).
- https://man7.org/linux/man-pages/man2/flock.2.html — advisory lock held
  across the whole read-modify-write, not just the media section.

## Fix candidates
1. Hold `_held_run_lock` for control-plane writes too (stop/pause take the
   lock non-blocking, fail with "commit in flight, retry").
2. Or add a `control_plane.json` sidecar the commit path merges instead of
   overwriting.
3. At minimum document the one-segment overrun in DESIGN §73.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.

## Progress (2026-10-07, group O)

- Decision (recorded): NO code fix — none is available within scope.
  The commit side of candidate 1 already holds `_held_run_lock` across
  the whole commit body including both `_write_state_preserving_control_plane`
  call sites, the single-threaded loop never writes control state
  concurrent with a commit, and there are no remaining writers to convert
  (no stop/pause CLI verbs exist; SIGINT is the in-process `_stop_flag`).
  A merge sidecar (candidate 2) would add a file + readers for a producer
  that does not exist. Took candidate 3 (the issue's own escape hatch).
- `Voyage/DESIGN.md`: one appended as-built paragraph after the 099 note
  (§73-control-plane-2026-10-07) — supersedes the stale "still open …
  `_set_status` and the TUI Stop button" line (both refer to deleted
  code), states the residual microsecond hand-edit window as LOST (not
  delayed), and records the no-sidecar rationale.
- `voyage/supervisor.py` (control-plane sections only): corrected the
  three stale `voyage stop` / `voyage pause` verb references
  (`_pause_requested`, `run_segments`, `_write_state_preserving_control_plane`
  docstrings) to describe the actual file-based control plane; the
  helper docstring now notes it runs under `_held_run_lock` on both call
  paths. No behavior change.
- New `tests/test_issue_223_control_plane.py` (4 tests: STOP/PAUSE present
  at read time win with counters advancing, no-request passthrough,
  missing-file fallback to fresh status).
- Verified in-container: `ruff check` clean, `ruff format --check` clean
  on own hunks (three flags at :837/:856/:960 are a concurrent agent's
  prewarm/mastering hunks — left untouched per §9), `mypy` strict clean
  on `supervisor.py`, scoped pytest 4/4 green; neighbors
  (`test_recovery`, `test_supervisor_hardening`, `test_commit_hardening`,
  `test_crash_matrix`, `test_failure_policy`) green.

## Resolution (2026-10-07)

DOCUMENTED (candidate 3, per the issue's own proviso). Present-at-read
requests are preserved and now pinned by tests; the only remaining loss
mode is an external hand-edit landing inside the microsecond
read-then-write window, which is accepted and written down in DESIGN §73.
Left open: nothing actionable — a future control-plane writer (if one is
ever reintroduced) must take `_held_run_lock` non-blocking and fail with
a retry message per candidate 1.

## Evaluation (2026-10-07, group O)

Re-verified live; shape confirmed, severity context narrowed:
- Read-then-write shape is real: `_write_state_preserving_control_plane`
  (`voyage/supervisor.py:2597-2614`; issue cites `:2386-2403`, shifted by
  later insertions) still reads `read_state`, merges REQUESTED, writes —
  with no lock and no CAS.
- BUT the blamed lock-free writers no longer exist: the two-verb CLI
  exposes only `configure` + `generate` parsers (`voyage/cli.py:240,330`)
  — no `stop`/`pause` verbs write `state.json` anywhere in `voyage/`
  (grep for status writes finds only supervisor-internal transitions in
  `run_segments` plus test fixtures). SIGINT arrives via the in-process
  `_stop_flag`, never via the file.
- The commit side of fix candidate 1 is already done: `commit_one_segment`
  (`:2907-2916`) holds `_held_run_lock` across the whole commit body
  including both `_write_state_preserving_control_plane` call sites
  (`:2723`, `:2835`), and the single-threaded loop never runs a control
  write concurrent with a commit. There are no remaining writers to
  convert to lock-taking, so candidate 1 has no missing half and
  candidate 2 (merge sidecar) would add a file + readers for a producer
  that does not exist.
- Residual window (real but producer-less in production): an external
  hand-edit of `state.json` landing between the helper's live read and
  its write-back is clobbered (microseconds per commit); a clobbered
  request is LOST, not merely delayed one segment (the issue's rationale
  overstates the recovery — the next boundary reads the clobbered
  RUNNING state). `tests/test_recovery.py:67-89` already documents the
  sibling race and pins only the between-runs behavior.
- Decision: code fix is not available within scope (nothing to convert,
  commit side already locked) → candidate 3: one DESIGN §73 paragraph
  (permitted single-docs-edit) + correct the stale `voyage stop` /
  `voyage pause` verb references in the touched control-plane docstrings
  + regression tests pinning the present-at-read preserve behavior.
