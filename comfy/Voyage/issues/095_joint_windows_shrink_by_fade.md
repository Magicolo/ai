# 095 — Take-joint windows shrink by the assemble fade (don't tile exactly)

- Status: resolved (fixed 2026-09-29: joint-compensating tail extension + explicit joint_fade + exact-tiling tests)
- Severity: low (correctness, inaudible at current scale)
- Area: final mix (`voyage/media.py: assemble_segment_audio` as used by
  `build_final_audio`)
- Rank rationale: systematic timeline inexactness in the shipped mix;
  bounded and absorbed today, but the blend's "timeline-exact" contract
  is only approximate at take joints.

## Technical description

`build_final_audio` assumes windows tile the timeline (each extended
half the overlap per side, pairwise blends subtract exactly one
overlap per joint). But a window containing a take joint is assembled
from contiguous slices joined with a crossfade of `fade`, so it comes
out `fade` short of its nominal range instead of tiling exactly.
Live evidence (poulah, 31 segments, overlap 0.4): window 11
(4.4 s range, joint at 45.375) nominal 4.133, window 21 nominal 4.0 —
0.667 s total absorbed into wider effective overlaps at those two
joints. No silence and no drift (the blend still spans every joint),
just 0.667 s of music content overlapped away instead of played.

## Why this is an issue

- The final mix is only exactly timeline-length when no take joint
  falls inside a window; caption changes (repaints) make joints more
  likely, so the most "experimental" runs drift most.
- At 0.667 s / 126 s it is inaudible — but it compounds with issue
  094's take variance in the same accounting, and neither is verified.

## Evidence

Instrumented `build_final_audio` run 2026-09-29 (every blend step
logged exp==actual — the machinery is exact; the inputs are short):

```
000011 slices (0.358 + 3.867, fade 0.179) -> 4.046 (range 4.4)
000021 slices (3.492 + 0.908, fade 0.4)   -> 4.000 (range 4.4)
FINAL: 125.288000 vs 126.083333 timeline
```

## Reproduction

Any run with a take joint inside a segment window (repaint, or seg0
take boundary at 45.375-style offsets): compare window durations
against nominal ranges in an instrumented `build_final_audio` run.

## Source references

- `voyage/media.py: assemble_segment_audio` (left-fold via
  `_blend_pair`; contiguous slices joined with a fade)
- `voyage/media.py: build_final_audio` (window loop assumes tiling)

## Resolution candidates

1. Extend joint slices by half the fade on each side before blending
   (mirror the window-extension trick) so assembled windows tile
   exactly; clamp slice bounds to take coverage.
2. Cheaper: assert per-window nominal length in `build_final_audio`
   and emit a `window_short` metric (visibility without behavior
   change).

## Investigation / progress / resolution log

- 2026-09-29: found while root-causing the poulah final-audio
  0.795 s shortfall (this accounts for 0.667 s; issue 094 covers the
  rest). Open: implement + test.
- 2026-09-29 (orchestrator): FIXED. `media._take_joint_fade` (single
  formula) + `assemble_segment_audio(..., joint_fade=)` override +
  `build_final_audio` extends each joint-window's tail piece by exactly
  fade*(joints) (clamped to the take file; deliberately NOT to the
  timeline — the extension restores nominal at most, so the last window
  is fixed where it needs it). Windows now tile nominally, so the final
  blend lands exactly on the timeline and `-shortest` stops trimming
  video. Tests: joint window == 4.0 nominal (was 3.8 pre-fix shape),
  jointless control exact, `joint_fade` unit. Gates green.
