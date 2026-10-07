# 306 — DESIGN §56 finalizer steps + command predate compression + two-verb CLI

Severity: MEDIUM (pass-2 DESIGN-docs drift sweep).

## Technical description

Steps 8 (stream-copy concat when compatible), 10 (`scale/pad to exact 768×432`), and the
`voyage finalize --run RUN_DIR --output final.mp4` command contradict the compression
change (every publish encodes `slow`/crf30, no stream-copy) and the two-verb CLI
(`configure`+`generate` only). The §56 as-built stack is self-contradictory
(veryfast/15 at `:3128` vs slow/30 at `:3129`).

## Rationale

Implementers/readers get three different finalize truths in one section; `media.py`'s own
docstring still says "crf 15 + veryfast."

## Live evidence

- Docs: `DESIGN.md:3100-3120` (steps 8/10 + command line `:3103`).
- Code: `voyage/media.py:431,444` (`FINALIZE_CRF_DEFAULT = 30`,
  `FINALIZE_PRESET_DEFAULT = "slow"`); `voyage/media.py:2153-2155` (`fast_path` stays
  False, stream-copy retired); `voyage/cli.py:239,329` (only `configure`/`generate`
  parsers); `voyage/media.py:859+` docstring still claims "crf 15 + veryfast."
- Command: `grep -n "add_parser" voyage/cli.py` → 2 verbs;
  `grep -n "FINALIZE_CRF_DEFAULT\|FINALIZE_PRESET" voyage/media.py`.

Repro: `voyage finalize --run X` → unrecognized verb (only `configure|generate` parse);
default finalize encodes (never copies).

## Source refs

As above.

## Online sources

- None (in-tree compression change + two-verb deletion are the anchors).

## Fix candidates

- Rewrite steps 8-10 as encode-always native publish; replace command with
  `configure`+`generate` equivalent; delete/supersede the `:3128` veryfast/15 as-built
  (keep `:3129`); fix `media.py` docstring defaults.

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.
