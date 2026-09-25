# 016 — Absolute paths persisted in ledger + metrics break run relocation

- Status: resolved (fixed 2026-09-25)
- Severity: major (portability / spurious FAILED + silent blend loss)
- Area: correctness — path persistence (`takes.jsonl`, `metrics.json`)
- Rank rationale: `cp -r` of a run (new mount, `/models` remap) is a normal
  operation that currently corrupts the next commit.

## Technical description

```python
take_file = audio_dir / f"{take.take_id}.wav"  # absolute when run_dir absolute
take.path = str(take_file)                     # stored verbatim in takes.jsonl
recovery_tape = tape  # worker absolute path stored in metrics.json
# cli.py:550
if isinstance(tape, str) and tape and not Path(tape).exists():
    errors.append(f"... references missing recovery checkpoint {tape}")
```

Moving/copying a run invalidates every take path and tape path. Next commit's
`slice_take(Path(serving.path))` → ffmpeg missing-file → `MediaError → FAILED`;
`build_final_audio` silently degrades to concat fallback (timeline still exact, but
blend lost with no metric distinguishing blend vs fallback); `validate_run`
reports missing-recovery errors on a perfectly intact relocated run.

## Why this is an issue

Copying or remounting a run — new disk, new mount, `/models` remap — is
routine operations work, yet it invalidates every persisted take path and
recovery-tape path at once. The next commit then fails on a missing file, the
final-audio assembly silently falls back from blend to concat with no
distinguishing metric, and `validate` cries missing-recovery on a perfectly
intact run. Portability is a durability property for a system designed to run
for weeks across machines, and run-relative storage costs nothing at the
supervisor boundary since the RPC wire can stay absolute. Operators migrating
runs — eventually everyone — pay with spurious FAILED states and silently
degraded audio.

## Evidence

Source quotes above (`voyage/supervisor.py:798,817,1152-1154`,
`voyage/audio/planner.py:29-36`, `voyage/cli.py:549-551`).

Re-verified 2026-09-25:

```
$ rg -n "take.path = str|take_file = audio_dir|recovery_tape = tape|recovery_tape: str" voyage/supervisor.py
764:        recovery_tape: str | None,
798:            take_file = audio_dir / f"{take.take_id}.wav"
817:            take.path = str(take_file)
1147:        recovery_tape: str | None = None
1154:                recovery_tape = tape
```

Absolute-path persistence confirmed at all three sites; no relativization.

## Reproduction

1. Commit 1 seg in `/tmp/a`; `mv /tmp/a /tmp/b`.
2. `validate` → `references missing recovery checkpoint /tmp/a/...`; next commit
   → audio-gap/`MediaError`.

## Source references

- Files/lines above.

## Resolution candidates

1. Persist **run-relative** paths (`audio/take_0000.wav`,
   `segments/N/recovery.pt`); resolve against `run_dir` at use.
2. Migrate legacy absolute entries on load (use as-is if it exists, else try
   `run_dir`-relative).
3. Keep wire payloads absolute (RPC necessity) but store relative — convert at the
   supervisor boundary. Also validates 006's containment check naturally.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Open: implement + relocation test (commit → move → validate → commit).
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  with live path-persistence output; refs verified current.
- 2026-09-25 (consumer-side resolution — PRODUCER HALF OPEN): shared
  convention helper `voyage/paths.py:resolve_stored_path` (existing path
  as-is, else `run_dir`-joined for relative entries; stored form stays
  run-relative POSIX); consumers wired: `AudioTake.resolved_path(run_dir)`
  (`voyage/audio/planner.py`), `media.build_final_audio` resolves every
  ledger path before serve/slice, `cli._check_segment_metrics` resolves
  the recovery tape (new optional `run_dir` arg, default preserves the
  legacy as-is check). Tests in `tests/test_run_relative_consumer.py`
  incl. commit → relativize → move → validate green. Concept-store paths
  needed no change (all derived from the store directory). HOOK FOR THE
  SUPERVISOR/PRODUCER TRACK (supervisor.py out of scope): persist
  run-relative at the boundary (`take.path`, `reference_audio`,
  `recovery_tape` in metrics) and resolve on load (use as-is if it
  exists, else `run_dir`-relative); `_ensure_audio_coverage` should call
  `serving.resolved_path(self._run_dir)` and resolve `current.path` for
  the repaint payload.
- 2026-09-25 (supervisor/producer half — DONE, status untouched for the
  media track): boundary now persists run-relative and resolves through
  the shared consumer convention — no private duplicate. `take.path`
  stores `audio/take_*.wav` via `_stored_relative`
  (`voyage/supervisor.py:266-277`); metrics `recovery_tape` stores
  `segments/NNNNNN/recovery.pt` (same helper, same site as the 006
  containment check); the RPC wire stays absolute (`output_path`,
  `reference_audio`, rebuild payload). Load sites use exactly the hooked
  calls: `serving.resolved_path(self._run_dir)` for slices and
  `current.resolved_path(self._run_dir)` for the repaint `reference_audio`
  (`voyage/supervisor.py:1047-1095`). An early private re-anchoring
  resolver was removed in favor of `paths.resolve_stored_path` to avoid
  convention drift. Relocation test
  (`test_relocated_run_continues_from_relative_paths` in
  `Voyage/tests/test_commit_hardening.py`): commit → assert relative ledger
  → `mv` → commit → `validate_run` clean; the media track's 9 consumer
  tests still pass against the new producer output (29/29 with the new
  hardening file). KNOWN GAP for the media track (paths.py is theirs): a
  *legacy absolute* entry that moved is returned stale-as-is by the shared
  helper (no re-anchor), so pre-relative runs still fail loudly after a
  move instead of healing — consider anchor-based re-anchoring in
  `resolve_stored_path` if that matters. Gates: full
  `Voyage/scripts/gates.sh` green (626 passed).
- 2026-09-25 (review): both halves landed + gap closed — producer stores
  run-relative (`supervisor._stored_relative`, wire stays absolute), consumers
  resolve via `paths.resolve_stored_path` / `AudioTake.resolved_path`, and the
  orchestrator added layout-anchored re-resolution for legacy absolute entries
  (`paths._LAYOUT_ANCHORS`: missing absolute re-anchors on first
  `segments/audio/novelty/logs` part when the target exists, else still fails
  loud). Tests: relocation suites both tracks + new
  `test_resolve_stored_path_reanchors_legacy_absolute`. Issue fully resolved.
