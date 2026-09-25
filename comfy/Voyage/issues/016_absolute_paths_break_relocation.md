# 016 — Absolute paths persisted in ledger + metrics break run relocation

- Status: open
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
