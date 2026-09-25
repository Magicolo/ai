# 062 — TUI `take_seconds` accepts `nan`/`inf`; `AudioConfig` validator accepts them too

- Status: resolved (fixed 2026-09-25, TUI track — both halves done)
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
- 2026-09-25 (fix, config part ONLY per batch split — `tui_state.py`
  untouched): FIXED the `AudioConfig` half — `non_negative`
  (`voyage/config.py:107-112`) and `overlap_cap_non_negative`
  (`voyage/config.py:130-133`) now reject non-finite via
  `math.isfinite` (`ValueError: must be a finite non-negative number`).
  `final_overlap_fraction`/`energy` already reject nan/inf through their
  range checks (verified live). Note: `DraftConfig.take_seconds`
  (`voyage/config.py:234-239`) shares the old pattern and is left for a
  later batch; it funnels into `AudioConfig` via `apply_draft_overrides`,
  so invalid values still fail at that layer. Tests:
  `test_audio_config_rejects_non_finite_take_seconds` +
  `test_audio_config_rejects_negative_take_seconds` in `tests/test_unit.py`.
  Gates: ruff + format + mypy strict clean on all scope files, 424 pytest
  passed in-container.
- Open (later batch): `field_errors` finiteness rejection in
  `voyage/tui_state.py:156-167` + nan/inf/negative tests per field.
- 2026-09-25 (fix, TUI track): relevance re-verified live (`nan`/`inf`
  passed `field_errors` with no error). Implemented `math.isfinite`
  rejection in `field_errors` (`voyage/tui_state.py`: `if not
  math.isfinite(take) or take <= 0`, message `positive finite number`).
  Plan-math half: `to_generate_namespace` already calls `validate`
  first and raises `ValueError` on any field error, so a nan/inf take
  can no longer reach planning math or the ACE payload — covered by
  test (`to_generate_namespace` raises on nan/inf). Duration needs no
  fix (`parse_duration`'s `\d+` regex rejects nan/inf with ValueError,
  verified live). Tests: `test_tui_state.py`
  `test_take_seconds_rejects_non_finite` (nan/inf/-inf/NAN/Infinity →
  field error + namespace raises), `test_take_seconds_rejects_non_positive`,
  `test_take_seconds_accepts_positive_finite`. Gates: `scripts/gates.sh`
  GREEN (ruff + format + mypy strict + 563 pytest).
