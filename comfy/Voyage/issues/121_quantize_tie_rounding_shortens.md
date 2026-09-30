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

## Progress log

- 2026-09-30 (Group B): re-verified live first (`voyage:latest`,
  CPU-only): `q(45, 18) = 36.0` (shortened 9 s), `q(45, 2) = q(45, 4) =
  44.0`. Bug premise confirmed — but the behavior is DELIBERATELY pinned
  by another pass: `tests/test_rhythm.py:66-67` expects `(45, 2) -> 44.0`
  ("22.5 segments rounds to 22") and `(45, 4) -> 44.0` ("11.25 segments
  rounds to 11"). Any no-shorten rule (ceil → 46/48, half-up → 46/44)
  breaks those pins, and the sole caller (`voyage/audio/planner.py:240`)
  plus the pin file are outside this group's file scope ("OWN FILES ONLY",
  "never undo another agent's work"). Changing the default here would
  trade one agent's green gates for another's red.
- Owned-file contribution instead: documented the tie rule in the
  `quantize_take_seconds` docstring (round-half-to-even, may plan short,
  094 clamp contains per-take damage, ceil/half-up preferred on
  renegotiation) + new `tests/test_beat_quantize_ties_121.py`
  CHARACTERIZATION (labeled as such per §12: pins exact-multiple, min-1,
  tie-down `45/18 -> 36.0`, tie-up `3.5 -> 4` cases) so a future rounding
  change must update the pins deliberately instead of silently.
- Gates (in-container): characterization + `test_rhythm` = 26 passed;
  `ruff check` + `ruff format --check` + `mypy voyage/audio/beat.py` clean.

## Resolution

- Verdict: CONFIRMED bug, fix DEFERRED (blocked, not folded — the defect is
  real). Files changed: `voyage/audio/beat.py` (docstring tie rule only,
  zero behavior change), `tests/test_beat_quantize_ties_121.py` (new
  characterization).
- DESIGN proposal (quoted text only, for the DESIGN owner): in §35, after
  the take-quantization description, add: "Take quantization uses
  round-half-to-even: exact-half segment ratios can plan short (a 45 s
  take on 18 s segments plans 36 s), chaining extra takes. Prefer ceil or
  round-half-up when the rhythm pins are renegotiated; until then the
  per-take clamp contains the damage."
- Residuals (exact handoff for the owning pass): switch
  `quantize_take_seconds` to `math.ceil` (or round-half-up) AND update
  `tests/test_rhythm.py:66-67` (`(45, 2) -> 46.0`, `(45, 4) -> 48.0` under
  ceil) plus `test_planner_quantizes_fresh_takes_to_segment_grid` (`44.0
  -> 46.0`) — needs the rhythm-test owner's sign-off, and consider the
  candidate-3 plan-site assertion (`take_seconds >> ahead_seconds`) in
  `voyage/audio/planner.py` at the same time.
