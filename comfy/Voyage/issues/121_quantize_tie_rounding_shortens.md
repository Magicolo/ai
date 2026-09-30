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

## Progress log (2026-09-30, tests-only pass — DECISION)

- Premise re-verified live (read-only, no host pip): `voyage/audio/beat.py`
  still `multiples = max(1, round(take_seconds / segment_seconds))`
  (round-half-to-even); sole production caller is
  `voyage/audio/planner.py:240` inside `_fresh_take` (only hit:
  `rg quantize_take_seconds voyage/` → `planner.py:238,240` + `beat.py`
  def; `__pycache__` hits ignored). No other production consumer depends
  on the tie direction.
- Pins re-verified live: `tests/test_rhythm.py:66-67` still expects
  `(45.0, 2.0, 44.0)` ("22.5 segments rounds to 22") and `(45.0, 4.0,
  44.0)` ("11.25 segments rounds to 11"); `test_rhythm.py:97-102`
  (`test_planner_quantizes_fresh_takes_to_segment_grid`) still expects
  `plan.take.duration == 44.0`; characterization file
  `tests/test_beat_quantize_ties_121.py` still pins tie-down `45/18 →
  36.0` (4 tests). Under ceil the first two pins become 46.0/48.0 and the
  planner pin becomes 46.0; under half-up (`floor(x+0.5)`) they become
  46.0/44.0 — either way at least one deliberately-pinned expectation plus
  the characterization file must change in the same commit as the
  `voyage/audio/beat.py` behavior edit.
- DECISION per the task contract: rhythm-owner-blocked, NOT implemented
  here. The behavior fix requires editing `voyage/audio/beat.py` (frozen
  `voyage/` scope) AND updating `tests/test_rhythm.py:66-67,102` +
  `tests/test_beat_quantize_ties_121.py` pins owned by the rhythm track —
  changing the default here would trade this pass's green for the rhythm
  owner's red (same collision the Group B log deferred). The pins are not
  pure throwaway characterization: the `test_rhythm.py:66` comment
  documents the rounding as intended ("rounds to 22"), and the planner pin
  encodes the 44.0 duration downstream consumers observe.
- No code change in this pass (tests/ owner cannot move first on a
  `voyage/` behavior + cross-owned pins). Gate evidence: n/a (no files
  changed).

## Resolution (2026-09-30, tests-only pass)

- Verdict: blocked (rhythm-owner-blocked with exact evidence above).
  Files changed: none. DESIGN proposals: none (Group B proposal stands).
- Residuals (exact handoff, rhythm/beat owner): switch
  `voyage/audio/beat.py:quantize_take_seconds` to `math.ceil` (never plan
  short) or round-half-up with the tie rule documented, AND update in the
  same commit: `tests/test_rhythm.py:66` `(45,2) 44.0 → 46.0`, `:67`
  `(45,4) 44.0 → 48.0` under ceil (46.0/44.0 under half-up — state the
  chosen rule), `test_rhythm.py:102` `44.0 → 46.0` (ceil) plus the 44.0
  literal at `:122`, and `tests/test_beat_quantize_ties_121.py:28-35`
  tie-down/tie-up pins; consider candidate-3 plan-site assertion
  (`take_seconds >> ahead_seconds`) in `voyage/audio/planner.py` at the
  same time. Needs the rhythm-test owner's sign-off — do not land the
  behavior half without the pin half.

## Progress log (2026-09-30, this pass — CHECK)

- Premise re-verified live in-container (`voyage:latest`, CPU-only, no
  host pip): `q(45,18)=36.0` (tie-down, 9 s short), `q(45,2)=44.0`,
  `q(45,4)=44.0`, `q(45,15)=45.0` (control); `voyage/audio/beat.py:110`
  still `round()`; sole production caller still
  `voyage/audio/planner.py:240` (`rg quantize_take_seconds voyage/` hits
  only `beat.py:88` def + `planner.py:238,240`).
- Pins re-verified live (unchanged — rhythm owner has NOT freed them):
  `tests/test_rhythm.py:66` `(45.0, 2.0, 44.0)` "22.5 segments rounds to
  22", `:67` `(45.0, 4.0, 44.0)` "11.25 segments rounds to 11",
  `:68` `(45.0, 5.04, 45.36)`, `:102`
  `plan.take.duration == pytest.approx(44.0)` plus the `44.0` literal at
  `:122`; `tests/test_beat_quantize_ties_121.py:32` tie-down `45/18 →
  36.0`, `:37` tie-up `3.5 → 4`. Recent history shows no freeing commit
  (`git log` tip `88496a4` batch 8; pins intact since batch 1 `b19a029`).
- Under ceil the `:66`/`:67` pins become 46.0/48.0 and the `:102`
  planner pin becomes 46.0; under half-up (`floor(x+0.5)`) they become
  46.0/44.0 — either way the behavior edit (`voyage/audio/beat.py`,
  outside own scope: only `planner.py` allowed, and the edit belongs in
  `beat.py`) must land with cross-owned pin updates in the same commit.
- No code change (own files only: `audio/beat.py` untouched,
  `planner.py` needs no edit without the `beat.py` switch). Gate
  evidence: n/a (no files changed); neighbor suite green in-container
  (`test_rhythm` + `test_beat_quantize_ties_121` pass as part of the 166
  run: 52 passed, 2 skipped — see issue 166 log for the full line).

## Resolution (2026-09-30, this pass)

- Verdict: blocked (rhythm-owner-blocked, exact pin cites above). Files
  changed: none (this issue file only). DESIGN proposals: none (Group B
  §35 proposal stands).
- Residuals (exact handoff, rhythm/beat owner): unchanged from the
  tests-only pass — switch `quantize_take_seconds` to `math.ceil` (never
  plan short) or round-half-up with the tie rule documented, AND update
  in the same commit `tests/test_rhythm.py:66,67,102` (+ `:122`
  literal) and `tests/test_beat_quantize_ties_121.py:28-37`, plus the
  candidate-3 plan-site assertion in `voyage/audio/planner.py`
  (`_fresh_take`, after `:240`). Needs the rhythm-test owner's sign-off.

## Progress log (2026-09-30, close-out pass — LANDED)

- Atomicity check first (`git status/diff`, root `/home/goulade/Projects/ai`):
  tree carries concurrent uncommitted hunks in OTHER files
  (`cli_observe.py` metrics extraction, `model_registry.py` JsonValue
  typing, `supervisor.py`, `augment_worker.py`, three issue files), but
  all four 121 target regions were foreign-hunk-free: `git diff HEAD --
  comfy/Voyage/voyage/audio/beat.py comfy/Voyage/voyage/audio/planner.py
  comfy/Voyage/tests/test_rhythm.py
  comfy/Voyage/tests/test_beat_quantize_ties_121.py` showed only this
  pass's edits (verified before landing). Contract condition met — landed.
- TDD red (in-container `voyage:latest`, CPU-only, no host pip): updated
  the pins first, watched fail against the old `round()` —
  `test_rhythm[45-2-46]`, `test_rhythm[45-4-48]`,
  `test_planner_quantizes_fresh_takes_to_segment_grid`,
  `test_exact_half_ratio_ceils_up`,
  `test_quantized_take_never_plans_short` = 5 failed, 22 passed.
- Fix: `beat.py:quantize_take_seconds` →
  `multiples = max(1, math.ceil(take_seconds / segment_seconds))`
  (`math` already imported) + docstring tie rule rewritten to ceil
  (never plans short; `45/18 = 2.5 → 3 → 54 s`); `planner.py:_fresh_take`
  gains the candidate-3 plan-site guard (`duration <= ahead_seconds`
  raises ValueError naming the swap cost + issue 121); pins updated in
  the SAME edit set: `test_rhythm.py:66` `(45,2) 44.0 → 46.0`
  ("22.5 ceils to 23"), `:67` `(45,4) 44.0 → 48.0` ("11.25 ceils to
  12"), `:102` `44.0 → 46.0`; `test_beat_quantize_ties_121.py` rewritten
  from CHARACTERIZATION to ceil pins (`45/18 → 54.0`, `3.5 → 4` same
  value new rule) + new `test_quantized_take_never_plans_short`
  (`q >= 45` on all three operating points). `:122` `AudioTake`
  hand-built `duration=44.0` left untouched (arbitrary ledger-roundtrip
  fixture, not a quantization pin — changing it would be churn).
- TDD green: `test_rhythm + test_beat_quantize_ties_121 +
  test_beat_properties + test_audio_planner + test_audio_take_ahead_guard`
  = 55 passed in-container. Scoped quality: `ruff check` clean,
  `ruff format --check` clean (4 files), `mypy
  voyage/audio/beat.py voyage/audio/planner.py` clean.
- Full `./Voyage/scripts/gates.sh`: 1714 passed, 10 skipped, 4 failed —
  all 4 foreign to this scope (verified by diff scope + failure
  signature): `test_tui_app::test_mid_run_progress_reaches_log_before_completion`
  (known TUI Pilot load-flake, shared-box), plus 3×
  `test_worker_perf_rank2` `test_047_*` asserting `augment_worker.py`
  source shapes (`half`/`gc.collect`/`_run_frame_batches`) while a
  concurrent agent holds a 391-line uncommitted `augment_worker.py`
  rewrite. No failure touches beat/planner/rhythm.

## Resolution (2026-09-30, close-out pass)

- Verdict: FIXED (landed, uncommitted per contract — no commit).
  Files changed: `voyage/audio/beat.py` (ceil + docstring),
  `voyage/audio/planner.py` (plan-site `duration > ahead_seconds`
  guard), `tests/test_rhythm.py` (3 pins),
  `tests/test_beat_quantize_ties_121.py` (ceil rewrite + no-shorten
  invariant). DESIGN proposals: none (Group B §35 proposal is now
  implemented — DESIGN owner may close it as done).
- Evidence: TDD red 5-failed → green 55-passed (lines above); scoped
  ruff + format + mypy clean; full gates 1714/4-foreign.
- Residuals: none for 121. Note for reviewers: the full-gate 4 failures
  belong to the concurrent `augment_worker.py` rewrite + TUI flake —
  do not revert this fix to chase them; re-run gates on an idle box.
