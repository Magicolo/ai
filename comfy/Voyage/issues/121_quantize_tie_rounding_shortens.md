# 121 — `quantize_take_seconds` uses banker's rounding: half-segment takes silently shorten, breaking coverage

- **Severity:** Low-Medium (beat math — deterministic coverage shortfall on exact-half ratios)
- **File:line:** `Voyage/voyage/audio/beat.py:61-76` (`multiples = max(1, round(take_seconds / segment_seconds))` at `:75`); consumer `Voyage/voyage/audio/planner.py:176-194` (`_fresh_take` duration)
- **Area:** workers-internals tail — `voyage/audio/beat.py` take quantization (below pass-1 coverage)

## Description

`round()` in Python is round-half-to-even: `round(2.5) == 2`, `round(3.5) == 4`. So quantization is asymmetric around halves — and the shorten direction is the dangerous one:

- `quantize_take_seconds(45.0, 18.0)` → ratio exactly `2.5` → `round → 2` → **36.0 s**: a requested 45 s take comes back 9 s short, silently.
- `quantize_take_seconds(45.0, 45.0/3.5)` → ratio `3.5` → `round → 4` → 51.4 s (lengthens — benign).

A shortened take understates coverage: the planner's `coverage_until` / audio-ahead math (`planner.py:112-116`, `take_covers_ahead` invariant in `config.py:340-357`) was reasoned about against the *requested* length. The supervisor does clamp the ledger to the probed file afterwards (issue 094 mechanism, `supervisor.py:1217-1224`), which contains the damage per-take — but the quantized duration is the *plan*, and planning short on exact halves means more chained takes (more GPU swaps) than the operator configured, with no diagnostic naming the rounding.

## Rationale

Take lengths are a coverage promise: "takes chain on segment-aligned boundaries" (beat.py:63-67). Rounding *down* on a tie breaks the promise in the one direction that creates gaps, while rounding *up* (ceiling, or round-half-up) can only over-cover — which the finalize slice walk already tolerates (slices are cut from coverage; excess is trimmed, shortfall fails loud per the 094 comment). The docstring says "snap to a whole number of segments" without specifying the tie behavior; the implementation picks the worst tie-break.

## Evidence (verified live 2026-09-30, host stdlib, `PYTHONPATH=Voyage`)

```
quantize(45,18) = 36.0        # 45/18 = 2.5 → round → 2 → 36 s (SHORTENED 9 s)
quantize(45,12.857) 3.5-ratio → 51.42857142857143   # 3.5 → 4 (lengthened)
```

## Repro

```bash
PYTHONPATH=Voyage python3 -c "
from voyage.audio.beat import quantize_take_seconds as q
print(q(45.0, 18.0))   # 36.0 — expect 45.0-adjacent (ceil → 54.0) or documented tie rule
print(q(45.0, 15.0))   # 45.0 exact — unaffected (control case)
"
```

## Fix candidates

1. Use `math.ceil` (never plan short — over-coverage is trimmed downstream) or round-half-up (`math.floor(x + 0.5)`) with the tie rule documented in the docstring.
2. Add example tests pinning `q(45, 18) >= 45` (no-shorten invariant) plus the exact-multiple control case.
3. Consider asserting the quantized result still satisfies `take_seconds >> ahead_seconds` at the plan site, so a future rounding change can't reintroduce per-segment GPU swaps silently.

## Refs

- `Voyage/voyage/audio/beat.py:61-76`; `Voyage/voyage/audio/planner.py:176-194`; `Voyage/voyage/config.py:340-357` (ahead invariant); `Voyage/voyage/supervisor.py:1217-1224` (094 clamp — contains, doesn't prevent).
- Adjacent, not overlapping: 120 (same module, tempo-ceiling axis); 100 (non-finite inputs — this file is finite-input rounding).
