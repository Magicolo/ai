# 144 — NaN measured metrics mislabel as WITHIN and yield no amendment

- Severity: MEDIUM (silent wrong label + dropped feedback; director steers on a lie)
- Area: director measured-context math
- Files (as-read 2026-09-30; no edits in these ranges):

## File:line

- `voyage/director.py:238-245` (`format_measured_context` band loop: `flag = BELOW if value < low else (ABOVE if value > high else WITHIN)`; `f"{value:.3f}"`)
- `voyage/prompts.py:89-107` (`feedback_amendments`: `motion > max` / `< min` / `complexity > max` / `drift < min` / `similarity < min` — all `None`-guarded, none finite-guarded)
- `voyage/config.py:305-310` (`AudioConfig.non_negative`: `not math.isfinite(value)` rejected — the finite-guard precedent this path lacks)

## Description

Measured visual metrics come from numpy/ffmpeg vision code — a `nan`/`inf` can escape (empty segment, zero-division, corrupt frame probe). Neither consumer guards finiteness: `format_measured_context` (`director.py:243`) maps any value that is neither `< low` nor `> high` to `WITHIN`, and every NaN comparison is `False`, so `nan` renders as `motion_energy=nan target=[..] WITHIN` — an explicit "all clear" for a missing measurement. `prompts.py:92-106` has the same hole: `nan > max` / `nan < min` are both `False`, so no amendment is emitted either. The director therefore receives a confident WITHIN plus zero corrective amendments for a value that means "unknown". `inf` is half-handled by accident (`inf > high` → ABOVE fires) but `-inf` → BELOW, and formatting prints `inf` raw.

## Rationale

- WITHIN is a positive claim ("the charter band holds") — unknown must never render as a positive claim.
- The codebase already decided this for config inputs (`config.py:308` rejects non-finite). Measured inputs deserve the same guard at the trust boundary (worker-reported floats → director prompt).
- Silent-drop + mislabel compose: the amendment path (the actual steering) stays quiet BECAUSE the label path lies.

## Live evidence

```
director.py:242  value = measured[name]
director.py:243  flag = "BELOW" if value < low else ("ABOVE" if value > high else "WITHIN")
director.py:244  lines.append(f"{name}={value:.3f} target=[{low:.2f},{high:.2f}] {flag}")
prompts.py:94   if motion > style.motion_energy_max: ...
prompts.py:96   elif motion < style.motion_energy_min: ...
# nan: both False → no amendment; director.py: both False → WITHIN
config.py:308   if not math.isfinite(value) or value < 0: raise ValueError(...)
```

`f"{float('nan'):.3f}"` → `'nan'` (no crash — the bug is semantic, not an exception).

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
from voyage.director import format_measured_context
from voyage.prompts import feedback_amendments
from voyage.config import StyleSpec
print(format_measured_context(StyleSpec(), {'motion_energy': float('nan')}))
print(feedback_amendments({'motion_energy': float('nan')}, StyleSpec()))
"
# Expect: 'motion_energy=nan ... WITHIN' and [] (no amendment).
```

## Fix candidates

- At the top of both functions: skip non-finite values (`math.isfinite` guard) — director.py either omits the line or emits `name=unknown (no measurement)`; prompts.py returns no amendment for that metric (explicit unknown, never WITHIN).
- Alternatively clamp/validate at the producer (`vision/metrics.py` boundary) so NaN never enters the dict. Belt-and-braces: guard both layers.
- Test: NaN/±inf for each of the six metrics → no WITHIN label, no misleading amendment.

## Refs

- `voyage/director.py:230-245`; `voyage/prompts.py:85-107`; `voyage/config.py:305-310` (finite-guard precedent); issue 100 (nan/inf config validation).
