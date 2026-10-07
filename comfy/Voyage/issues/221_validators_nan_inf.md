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

## Evaluation (2026-10-07)
- Re-read `voyage/workers/_validators.py:31-66` live: all five range-only
  validators (`sample_rate`/`channels`/`geometry`/`fps`/`frame_count`)
  still test only `<= 0` / `not in (1, 2)` — `nan`/`inf` pass (`nan <= 0`
  and `inf <= 0` are both `False`), `True` passes as 1 everywhere
  (`True == 1`, `True in (1, 2)`). `validate_energy`/`validate_duration_seconds`
  already demand `math.isfinite`. NOT stale.
- Live repro in-container (`voyage:latest`): `validate_sample_rate(nan)`,
  `validate_fps(inf)`, `validate_frame_count(True)`,
  `validate_geometry(nan, 512)` all PASSED (no raise) — bug confirmed.
- `loop.py:122` maps `(ValueError, KeyError, TypeError)` to
  `INVALID_PAYLOAD`, so the new `ValueError` for `str` inputs stays in
  the same taxonomy as the old raw `TypeError` (no behavior change there).

## Progress log (2026-10-07)
- Added `isinstance(v, bool)` rejection + `isinstance(v, (int, float))` +
  `math.isfinite` guards to `validate_sample_rate`/`validate_geometry`/
  `validate_fps`/`validate_frame_count`, and a `bool` guard to
  `validate_channels` (`nan`/`inf` already fail its `not in (1, 2)`;
  `True == 1` was the gap), all mirroring `validate_energy`.
- Added `test_shared_validators_reject_non_finite_and_bool_221` to
  `tests/test_worker_validators_unified_084.py` (NaN/±inf/True/False pins
  across all five validators + both geometry axes).

## Resolution (2026-10-07)
- RESOLVED. Files: `voyage/workers/_validators.py`,
  `tests/test_worker_validators_unified_084.py`. Scoped + neighbor pytest
  green in-container; ruff + format + mypy strict clean on touched modules.
  Nothing left open.
