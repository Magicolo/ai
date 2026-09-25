# 062 — TUI `take_seconds` accepts `nan`/`inf`; `AudioConfig` validator accepts them too

- Status: open
- Severity: medium (non-finite take length flows into planning + ACE payload)
- Area: TUI + config validation — `voyage/tui_state.py:156-167`,
  `voyage/config.py:107-112`
- Rank rationale: pass-2 finding; distinct field/gate from 039's
  `beats_for_segment` hole (`take_seconds` text input + pydantic validator).

## Technical description

```python
take = float(state.take_seconds.strip())
...
if take <= 0:  # nan<=0 is False → passes; inf>0 → passes
```

```python
@field_validator("take_seconds", "ahead_seconds", "crossfade_seconds")
def non_negative(cls, value: float):
    if value < 0:  # nan<0 is False → passes
```

## Why this is an issue

`nan` slips past every comparison (`nan <= 0` is `False`) and `inf` passes
positivity, so a non-finite take length flows from the TUI text field through
planning math into the ACE payload — producing garbage schedules or downstream
crashes far from the input that caused them. The pydantic validator shares the
hole, so neither layer catches what a single finiteness check would.

## Evidence (live probes by pass-2 sweep)

```
{'take_seconds': 'nan'} -> output='output/voyage' take=nan dur=5.0   (no field error)
{'take_seconds': 'inf'} -> take=inf                                  (no field error)
AudioConfig(take_seconds=nan) -> ACCEPTED nan
AudioConfig(take_seconds=inf) -> ACCEPTED inf
```

## Reproduction

`to_generate_namespace(GenerateFormState(style='x', take_seconds='nan'))`
succeeds with `take=nan`; `AudioConfig(take_seconds=float('nan'))` constructs.

## Source references

- Files/lines above.

## Resolution candidates

`if not math.isfinite(take)` rejection in `field_errors` + `non_negative`
validator (`if value < 0 or not math.isfinite(value)`). Consider one shared
`_require_finite` helper with 039's beat-math fix.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`tui_state.py` take block `:156-167`,
  `config.py:107-112` `non_negative` — both match).
- Open: implement + tests (nan/inf/negative per field).
