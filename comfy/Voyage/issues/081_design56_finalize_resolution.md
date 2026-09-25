# 081 — DESIGN §§56–57 vs code: finalize keeps generation resolution instead of normalizing to 768×432

- Status: open
- Severity: low (spec drift — deliberate as-built deviation without a DESIGN
  note)
- Area: spec/code drift — DESIGN §56 (`DESIGN.md:2857`, step 10), §57
  (`DESIGN.md:2885`), `voyage/cli.py:640-655`
- Rank rationale: pass-2 DESIGN finding; §§56–57 not covered by 047's
  README/BACKENDS/MODELS drift.

## Technical description

```python
finalize_run(run_dir, output,
    # Finalize keeps the generation resolution (no downscale): the
    # run config snapshot carries what the segments rendered at,
    width=config.video.width, height=config.video.height, ...
```

Spec still says: step 10 "scale/pad to exact 768×432" and §57 "The final output
target is 768×432 … 24 fps 16:9". §57 reads as normative while 768×512 runs
finalize natively.

## Why this is an issue

The code deliberately finalizes at generation resolution (keeping full detail,
letting old runs refinalize natively), but the spec still reads as normative
768×432 — so a reader cannot tell deliberate as-built deviation from drift.
Without an as-built note, a future "fix to spec" would reintroduce the
downscale/pillar loss the deviation was made to avoid.

## Evidence

Code side (re-run 2026-09-25):

```
$ rg -n "keeps the generation resolution" Voyage/voyage/cli.py
643:            # Finalize keeps the generation resolution (no downscale): the
$ rg -n "scale/pad to exact|768.432" Voyage/DESIGN.md
485:- 768×432 and the closest model-native resolution if 768×432 is internally binned or padded.
...
```

Spec side: §56 step 10 "scale/pad to exact 768×432 if needed" and §57 "The
final output target is 768×432 … 24 fps 16:9" still read as normative while
768×512 runs finalize natively (`cli.py:640-655`).

## Reproduction

Read the two sites.

## Source references

- Files/lines above.

## Resolution candidates

Append as-built note to §§56–57 (generation-resolution finalize; 768×432 is
legacy), as done for §118.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI/DESIGN sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`cli.py:640-655` finalize keeps generation resolution;
  DESIGN §56 step 10 + §57 768×432 normative text — both present); re-ran
  `rg` probes (pasted above).
- Open: annotate DESIGN.
