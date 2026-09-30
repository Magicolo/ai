# 188 — `finalize --skip-bad` silently disables the numbering-gap check (138 misdescribes it as strict)

- Severity: LOW-MEDIUM
- Area: media finalizer — error-policy inconsistency
- Files (as-read 2026-09-30):
  - `voyage/media.py:934-940` (numbering-gap check, guarded by `if not settings.skip_bad`)
  - `voyage/media.py:941-950` (the `_verify_segment` skip loop — the only lenient leg)
  - `Voyage/issues/138_finalize_skip_bad_three_legs.md:65` ("numbering gap stays strict")

## Description

The numbering-contiguity check is wrapped in `if not settings.skip_bad:`, so under
`--skip-bad` the check does not run at all — mis-numbered segments are accepted
silently, with no `finalize: skipping …` line and no warning:

```python
# media.py:934-940 (as-read)
if not settings.skip_bad:
    for position, segment in enumerate(committed):
        if segment.name != f"{position:06d}":
            raise MediaError(
                f"segment numbering gap: expected {position:06d}, found {segment.name}"
            )
```

Issue 138 documents this line as "numbering gap stays strict" — the code says the
opposite: strict mode raises, lenient mode skips the check entirely rather than
treating the gap as a skippable defect. So `--skip-bad` has two different meanings
in one function: checksum/metrics/alignment failures are skipped *with a warning*
(leg 2), while numbering gaps are ignored *without even a warning*. An operator
recovering a run with a missing segment (the canonical `--skip-bad` case: DONE
dirs `000000, 000002`) gets a final video whose timeline silently elides the gap —
`_segment_timeline` just concatenates whatever `usable` holds in sorted order —
with nothing in stdout or metrics saying a segment number is missing.

## Rationale

Fail-loud vs fail-lenient must be explicit per leg. Silent acceptance is the worst
of both: strict users get no protection signal, lenient users get no record of
what shipped. The timeline math downstream (`_segment_timeline`, SFX bounds) keys
off sorted order, not names, so the gap is invisible in the artifact too.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '934,950p' voyage/media.py` — gap check inside `if not settings.skip_bad:`;
  the skip loop below only wraps `_verify_segment`, never the numbering check.
- `sed -n '363,384p' voyage/media.py` (`_segment_timeline`) — iterates `usable` in
  given order with a running cursor; nothing references segment names, so a gap
  leaves no trace in the mix.
- 138's own evidence block quotes `:935` with the comment "numbering gap stays
  strict" — inverted vs the live `if not` guard.

## Repro

1. Build a run with committed segments `000000` and `000002` (delete `000001`).
2. `finalize_run(run_dir, out, skip_bad=False)` → `MediaError: segment numbering gap:
   expected 000001, found 000002`.
3. `finalize_run(run_dir, out, skip_bad=True)` → succeeds with zero mention of the
   missing `000001` in stdout; output duration equals the two segments back-to-back.

## Fix candidates

1. (Preferred) Treat the gap as a fourth skippable leg: under `skip_bad`, emit
   `finalize: skipping …`-style warning naming the expected vs found segment and
   continue (segments are already sorted, so no other change needed); keep the
   strict raise when `skip_bad=False`.
2. Log skipped/gapped segments as structured metric events (same ask as 138's fix
   candidate 3) so `status`/`scoreboard` can show what shipped.
3. Correct 138's "stays strict" line when fixing (cross-reference, do not edit
   138's file — index annotation only).
4. Tests: gap + `skip_bad=True` → finalizes with a recorded warning; gap +
   `skip_bad=False` → raises (already pinned implicitly, pin explicitly).

## Refs

- In-tree: `voyage/media.py:921-952` (`committed` discovery, gap check, skip loop);
  `voyage/media.py:363-384` (`_segment_timeline` order-dependence).
- Not-a-duplicate: 138 files legs 1 (existence probe), 2 (`_verify_segment`), 3
  (post-assembly `validate_video`) and explicitly mislabels this fourth leg as
  strict — this file covers only the numbering-gap leg and the misdescription.
  109 is CLI plumbing for the flag; 098 is orphan-scan coverage, not numbering.
