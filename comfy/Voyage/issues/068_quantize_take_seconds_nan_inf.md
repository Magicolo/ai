# 068 — `quantize_take_seconds` nan/inf escapes the documented `ValueError` contract

- Status: open
- Severity: medium (bare `round()` internals leak; `inf` segment silently yields
  `inf` take)
- Area: audio beat math — `voyage/audio/beat.py:45-58`,
  `Voyage/tests/test_rhythm.py:36-42,61-65`
- Rank rationale: pass-2 finding; sibling of 039's `beats_for_segment` hole with
  a third variant (`inf` → `inf` accepted downstream).

## Technical description

Guards only `<= 0`. `round(nan)` raises bare `ValueError: cannot convert float
NaN to integer` (leaks `round()` internals, no "take length" message);
`round(inf)` raises uncaught `OverflowError`; `quantize_take_seconds(45.0, inf)`
silently returns `inf`. Tests reject only `0.0`/`-2.0`/`0` beats — no nan/inf
anywhere in `test_rhythm.py`.

## Why this is an issue

The function documents a `ValueError` contract but leaks `round()` internals
instead — a bare "cannot convert float NaN" message for `nan`, an uncaught
`OverflowError` for infinite take length — while `(45.0, inf)` silently returns
`inf` and flows downstream as a valid take. Callers catching `ValueError` get
the wrong exception class on one path and no exception at all on another.

## Evidence (code experiments, verified by orchestrator 2026-09-25)

```
$ PYTHONPATH=Voyage python3 -c "...quantize_take_seconds(45.0,float('inf'))..."
inf
```

(`nan` → ValueError-from-`round`, `inf` → OverflowError per sweep probes.)

Re-verified 2026-09-25 (`PYTHONPATH=Voyage`):

```
$ python3 -c "from voyage.audio.beat import quantize_take_seconds; print(quantize_take_seconds(45.0, float('inf')))"
inf
$ python3 -c "from voyage.audio.beat import quantize_take_seconds; print(quantize_take_seconds(float('nan'), 4))"
ValueError: cannot convert float NaN to integer
```

## Reproduction

`quantize_take_seconds(float('nan'), 4)`, `(45.0, float('inf'))`,
`(float('inf'), 4)`.

## Source references

- Files/lines above.

## Resolution candidates

`if not math.isfinite(...)` → `ValueError` in both functions (or one shared
`_require_finite` helper with 039/062); extend the existing
`rejects_non_positive` tests with nan/inf cases.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep; `inf` case re-verified live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`beat.py:45-58`, `test_rhythm.py:36-42,61-65` — all match);
  re-ran nan/inf probes (outputs pasted above).
- Open: implement + tests.
