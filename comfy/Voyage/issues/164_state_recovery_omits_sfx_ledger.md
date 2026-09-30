# 164 — STATE_AND_RECOVERY invariants omit the SFX ledger (validator exists, doc does not)

- **Severity:** LOW (docs-only — `validate` checks the SFX ledger, the invariants list never says so)
- **File:line:** `Voyage/docs/STATE_AND_RECOVERY.md:3-26` (invariants 1-8, no SFX) vs `Voyage/voyage/sfx_finalize.py:51` (`SFX_LEDGER_NAME`) + `Voyage/voyage/sfx_finalize.py:213-229` (`validate_sfx_ledger`) + `Voyage/voyage/cli.py:883-892` (committed-DONE scan) + `Voyage/voyage/cli.py:923-929` (validate wires the SFX ledger check)
- **Area:** state-docs gap (SFX pass is validated but not specified)

## Description

`docs/STATE_AND_RECOVERY.md:3-26` lists eight commit invariants
(DONE-gating, contiguous numbering, state-follows-artifacts,
checksum recomputation, atomic writes, frame-range + A/V alignment,
no orphan partials, tape numerics). None mentions the SFX ledger —
yet the code treats it as validated state:

```python
# sfx_finalize.py:51-52
SFX_LEDGER_NAME = "sfx.jsonl"
SFX_STEMS_DIRNAME = "sfx"
```

```python
# sfx_finalize.py:213-217
def validate_sfx_ledger(run_dir: Path, timeline_seconds: float) -> list[str]:
    """Read-only SFX checks: files exist, windows tile the timeline.
    No ledger (SFX never ran / old run) is clean — the pass is optional.
    """
```

and `validate_run` (`cli.py:923-929`) extends its errors with it on
every non-empty timeline:

```python
# cli.py:923-929
fps = state.fps if isinstance(state.fps, int) and state.fps > 0 else 24
timeline = state.timeline_frames / fps
if timeline > 0.0:
    from voyage.sfx_finalize import validate_sfx_ledger
    errors.extend(validate_sfx_ledger(run_dir, timeline))
```

So the implemented contract is: stems under `audio/sfx/` + ledger
`audio/sfx/sfx.jsonl` tiling the timeline, absent-ledger-is-clean
(optional pass), checked read-only by validate — while the doc's
contract stops at invariant 8 with no SFX line, no pointer to the
ledger path, and no statement of the absent-is-clean rule. A reader
implementing "all enforced by `voyage validate`" from the doc alone
would miss the ninth check (and, worse, might "fix" a ledger-less
old run that validate correctly accepts).

The surrounding validate machinery the doc *does* describe is
accurate: `cli.py:883-892` scans `segments/` for DONE dirs and
compares against `state.committed_segments` exactly as invariants
1-3 promise.

## Rationale

The invariants list is normative ("all enforced by `voyage
validate`, testable via `validate_run`") — an enforced check missing
from the list is a spec hole in both directions: auditors cannot
verify completeness, and future refactors can drop the
`validate_sfx_ledger` call with no doc test failing. One invariant
line + the absent-is-clean rule closes it.

## Live evidence

- `sed -n '3,26p' docs/STATE_AND_RECOVERY.md` — invariants 1-8;
  `grep -n "sfx\|SFX\|ledger" docs/STATE_AND_RECOVERY.md` → no hits
  (outside this file's future line).
- `sed -n '51,58p' voyage/sfx_finalize.py` — ledger/stems names +
  worker-module map.
- `sed -n '213,229p' voyage/sfx_finalize.py` — read-only checker:
  missing ledger → `[]`, else file-existence + timeline tiling.
- `sed -n '883,929p' voyage/cli.py` — DONE scan (`:883-892`) then
  the SFX extension (`:923-929`) gated on `timeline > 0.0`.
- Overlap check: 101 is the ledger-fsync gap (durability of writes,
  not doc coverage); 054 is the ledger race (concurrency, not docs).
  Neither names the invariants-list omission.

## Repro

Static: diff the eight invariants against the `validate_run` body —
`validate_sfx_ledger` (`cli.py:926`) has no corresponding invariant
line. Dynamic: validate an SFX-finalized run with a truncated ledger
→ error names `sfx ledger …`, a string no invariant predicts; delete
the whole `audio/sfx/` dir on an old run → clean, a rule no
invariant states.

## Fix candidates

1. Add invariant 9: "**SFX ledger tiles the timeline.**
   `audio/sfx/sfx.jsonl` + stems tile `[0, timeline)` (checked
   read-only by `validate_sfx_ledger`); absent ledger is clean (SFX
   is an optional finalize-time pass / pre-SFX runs)."
2. Point at the paths (`SFX_STEMS_DIRNAME`/`SFX_LEDGER_NAME`,
   `sfx_finalize.py:51`) so the invariant is locatable.
3. Test: docs-vs-validate assertion that every `errors.extend(...)`
   source in `validate_run` has a named invariant (guards the next
   added check the same way).

## Refs

- `Voyage/docs/STATE_AND_RECOVERY.md:3-35`; `Voyage/voyage/sfx_finalize.py:48-62,141-229`;
  `Voyage/voyage/cli.py:883-929`.
- Adjacent, not overlapping: 101 (ledger fsync durability); 054
  (ledger race); 098 (orphan-scan gaps — invariant 7's checker, not
  the missing invariant 9).
