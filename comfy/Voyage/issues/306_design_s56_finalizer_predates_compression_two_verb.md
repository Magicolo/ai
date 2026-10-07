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

## Evaluation (2026-10-07)

Claim CONFIRMED live. `voyage/media.py:431` (`FINALIZE_CRF_DEFAULT = 30`)
+ `:444` (`FINALIZE_PRESET_DEFAULT = "slow"`) + `:2154-2155`
(`fast_path` False, stream-copy retired); `voyage/cli.py:239,329` are
the only two `add_parser` calls (`configure` + `generate`); both
`finalize_run` docstring (`:892`, "crf 15 + veryfast") and
`FinalizeOptions` docstring (`:532`, same stale defaults vs live
`:557-558` = 30/slow) contradicted the code. Step 6 (0.6 s A/V budget)
verified still live (`:3124-3126` as-built + `av_drift_seconds`) — left
untouched. Remaining `voyage finalize` hits elsewhere in DESIGN.md
(:3196 verb list, :3307 `voyage validate/finalize` surface, :4939/:5731/
:6162 old runbooks, :8704/:8727 §140 logs) are other sections' drift or
immutable dated logs — out of scope (§56 only), some overlapping issue
218's deleted-verb territory. Follow-up noted: the §57 as-built
(`:3161-3163`) still says "Steps 10 (§56) … read as normative 768×432
legacy" — now stale for step 10 only (rewritten to presentation-box
language); needs a §57-scoped touch.

## Progress log

- 2026-10-07: §56 command → `configure`+`generate`; steps 8/9/10 →
  encode-always native publish (single libx264 encode, presentation box
  from `plan_augmentation`); `:3128` veryfast/15 as-built marked
  pre-compression/superseded (kept, `:3129` compression as-built kept as
  authority); fixed both `media.py` docstrings (30 + slow). No code
  behavior touched (docstring-only).

## Resolution (2026-10-07)

RESOLVED docs-only (+ 2 docstring hunks). Files: `Voyage/DESIGN.md`
(§56 only), `Voyage/voyage/media.py` (docstring hunks at `:531-533`
and `:892` only).
Verify: `grep -n "voyage finalize --run\|stream-copy concat when\|768×432 if needed" Voyage/DESIGN.md`
→ no hits outside the superseded `:3134` as-built quote;
`ruff format --check` + `ruff check` on `media.py` clean.
