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

## Evaluation (2026-10-07)

Claim CONFIRMED live. `voyage/augment.py:136-158`
(`interpolated_chunk_frame_count`: "no boundary pair is skipped") +
`:161-187` (`chunk_windows`: stride `size - 1`, exact unchunked
`(n-1)*m+1` total) implement the 2026-10-07 overlap fix; the filed refs
(`:136-183`) still cover both functions. `docs/AUGMENT.md:104-108`
verbatim described the retired lossy tradeoff. `size == 1` legacy
exception confirmed in code (`:146-148,174`) and preserved in the fix.

## Progress log

- 2026-10-07: rewrote the paragraph to overlapping-windows + exact
  totals + `size == 1` legacy-lossy exception, citing
  `interpolated_chunk_frame_count` (`:136-158`) / `chunk_windows`
  (`:161-187`). No code touched.

## Resolution (2026-10-07)

RESOLVED docs-only. File: `Voyage/docs/AUGMENT.md`.
Verify: `grep -n "tiny hitch\|NOT sum" Voyage/docs/AUGMENT.md` → no hits;
chunk size now documented as memory-vs-throughput only.
