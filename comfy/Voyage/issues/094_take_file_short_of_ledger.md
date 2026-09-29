# 094 — Rendered take files run short of their ledger duration (ACE variance)

- Status: resolved (fixed 2026-09-29: commit-side probe+clamp with take_short metric + bounded coverage re-plan + tests)
- Severity: minor (operability / accounting drift)
- Area: audio takes (`voyage/audio/acestep.py` render → `audio/takes.jsonl` ledger)
- Rank rationale: silent ledger/file skew that the window slicer absorbs
  today, but nothing verifies it — a larger variance would open a real
  coverage gap.

## Technical description

ACE-Step renders are not sample-exact vs the requested duration. Live
evidence (poulah, 2026-09-29): `take_0000` and `take_0007` ledger
`duration: 45.375` but the committed WAV files probe at `45.2`
(−0.175 s each); the eleven 44.0 s takes are exact. The window slicer
(`build_final_audio`) asks `slice_take` for ledger-shaped pieces and
ffmpeg silently clamps to EOF, so the windows came out short (w10
4.358 vs 4.4, w11 slice 0.358 vs 0.535) and the final blend absorbed
the rest in slightly wider crossfades — no silence, no failure, but
the ledger now overstates coverage by 0.175 s per affected take.

## Why this is an issue

- The ledger is the planning truth (`take.covers_until()` drives the
  slow loop and every re-slice). Unverified durations let real gaps
  hide behind ledger math until a slice lands fully past EOF (then
  `piece_end <= cursor` degrades the whole finalize to the hard-splice
  fallback — a cliff, not a slope).
- 0.175 s is ACE variance today; nothing bounds it tomorrow.

## Evidence

```
take_0000 ledger 45.375 file 45.2 delta -0.175
take_0007 ledger 45.375 file 45.2 delta -0.175
(take_0001..0006, 0008..0012: delta 0.0)
```

Probed 2026-09-29 in `voyage:latest` against
`Voyage/output/poulah/audio/take_*.wav` vs `takes.jsonl`.

## Reproduction

Any ACE run whose render comes back short (45.375 s takes show it;
44.0 s takes don't): compare `ffprobe` duration of the committed take
against its ledger line.

## Source references

- `voyage/audio/acestep.py: render_take` (no duration check on the
  rendered file before the ledger append)
- `voyage/supervisor.py: _ensure_audio_coverage` (records `take.path`
  and appends without probing)
- `voyage/media.py: build_final_audio` (slice loop clamps silently)

## Resolution candidates

1. Probe the converted take at commit; on shortfall either re-render
   once or clamp the ledger `duration` to the file (plus a
   `take_short` metric so it stays visible).
2. Cheaper: assert in `build_final_audio` that every window reached
   its nominal length (fail loud instead of absorbing).

## Investigation / progress / resolution log

- 2026-09-29: found while root-causing the poulah final-audio
  0.795 s shortfall (take variance accounts for ~0.13 s of window
  deficit; the rest is issue 095). Open: implement + test.
- 2026-09-29 (orchestrator): FIXED. `media.probed_take_seconds` (probe +
  MediaError on empty) + supervisor `_ensure_audio_coverage` probes every
  rendered take, clamps `take.duration` to the file, and logs
  `take_file_seconds`/`take_short_seconds` on the `take_rendered` event;
  a bounded 3-attempt coverage loop re-plans when the clamp leaves the
  segment uncovered (same chained-take machinery, slice walk still fails
  loud past the bound). Tests: `tests/test_audio_accounting.py`
  (probe exact/reject, commit-clamp via stubbed probe on a real fake
  commit asserting ledger 45.x→1.0). Gates green.
