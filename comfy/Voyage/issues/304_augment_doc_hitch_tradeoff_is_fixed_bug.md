# 304 — AUGMENT.md chunk-hitch tradeoff is the fixed bug, not current behavior

Severity: MEDIUM-HIGH (pass-2 DESIGN-docs drift sweep).

## Technical description

`docs/AUGMENT.md:104-108` says chunk outputs do NOT sum to `(n-1)*m+1`, one boundary pair
skipped per joint ("tiny hitch per 32 frames"). Since the 2026-10-07 intra-segment jump
fix, windows overlap by one frame and chunked output sums exactly.

## Rationale

Users tuning chunk size for "fewer hitches" or diagnosing jumps are acting on a retired
tradeoff; the §140 entry itself calls the old docstring "the bug, not a tradeoff."

## Live evidence

- Docs: `docs/AUGMENT.md:104-108` verbatim hitch paragraph.
- Code: `voyage/augment.py:136-183` — `interpolated_chunk_frame_count` ("no boundary pair
  is skipped") + `chunk_windows` ("tiling … with overlap", stride `size-1`);
  `DESIGN.md:8783+` intra-segment-jump entry confirms.
- Command: `sed -n '104,110p' docs/AUGMENT.md` vs `sed -n '136,183p' voyage/augment.py`.

Repro: `chunk_windows(64, 32)` yields overlapping `(start,count)`; per-chunk
`interpolated_chunk_frame_count` sums to exact `(n-1)*m+1` (64f/m4 = 253).

## Source refs

`docs/AUGMENT.md:104-108`; `voyage/augment.py:136-183`; `DESIGN.md:8783+`.

## Online sources

- None (in-tree fix entry is the anchor).

## Fix candidates

- Rewrite the paragraph to overlapping-windows + exact totals; keep the `size==1`
  legacy-lossy exception (code preserves it); cite `interpolated_chunk_frame_count`/
  `chunk_windows`.

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.
