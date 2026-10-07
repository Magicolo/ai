# 221 — `_validators` accept `NaN`/`inf` (and `bool`-as-int slips past range checks) (MEDIUM)

## Technical description
`validate_sample_rate/channels/geometry/fps/frame_count`
(`voyage/workers/_validators.py:31-66`) test only `<= 0`. Since `nan <= 0`
and `inf <= 0` are both `False`, non-finite values pass, while
`validate_energy`/`validate_duration_seconds` correctly demand
`math.isfinite`. No `isinstance` check either (`"48" <= 0` raises raw
`TypeError` → `INVALID_PAYLOAD`, ok, but `True` passes as 1 where
`checked_request` already rejected it — defense-in-depth gap if validators
are called directly, as GPU workers do).

## Rationale
Bad numbers fail deep inside ffmpeg/torch instead of at the op boundary as
`INVALID_PAYLOAD` (fatal, no GPU load) — the exact failure the validators
exist to prevent (docstring: "before any ffmpeg side effect").

## Live evidence (host, stdlib-only)
```
command: PYTHONPATH=Voyage python3 -c "..."
output:
validate_sample_rate nan PASSED (no raise) BUG
validate_sample_rate inf PASSED (no raise) BUG
validate_fps nan PASSED (no raise) BUG
validate_energy nan raised ValueError
validate_duration_seconds nan raised ValueError
```
(`validate_geometry(nan,512)` also passes.)

## Repro
Call any of the five range-only validators with `float("nan")` /
`float("inf")`.

## Source refs
- `Voyage/voyage/workers/_validators.py:31-66` vs `:69-88`

## Online sources
- https://docs.python.org/3/library/math.html#math.isfinite — `nan`/`inf`
  unordered comparisons.
- https://pydantic-docs.helpmanual.io/usage/validators/ — validate at the
  boundary (mirrors `checked_request` doctrine).

## Fix candidates
Add `math.isfinite` + `isinstance(v,(int,float)) and not
isinstance(v,bool)` to all five, mirroring `validate_energy`; add
NaN/inf/bool parametrized pins.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.
