# 180 — `feedback_amendments` steers on 4 of 6 §43 metrics: `palette_distance` + `scene_boundary_strength` never produce amendments

- **Severity:** MEDIUM-LOW (director sees two metrics it can never act on; palette blowout / scene-cut instability get measured, labeled, then ignored)
- **Track:** C (prompts/director tails — below 026/144/118)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; no edits in these ranges)

## File:line (live-verified)

- `voyage/prompts.py:82-107` (`feedback_amendments`: branches for `motion_energy`, `visual_complexity`, `semantic_change_rate`, `style_similarity` only — no `palette_distance`, no `scene_boundary_strength`)
- `voyage/scoreboard.py:25-32` (`METRIC_KEYS`: the six §43 metrics, in order — the contract the feedback path half-honors)
- `voyage/director.py:230-237` (`format_measured_context` bands: all six rendered, with `palette_distance`/`scene_boundary_strength` on hardcoded `[0.00,0.30]` bands)
- `voyage/models.py:67-82` (`StyleSpec`: charter bands for motion/complexity/drift/similarity only — no palette/scene fields, so the feedback function has nothing to compare them against)

## Description

The MEASURED block carries six deterministic metrics. `format_measured_context` renders all six (the director prompt shows `palette_distance=0.990 ... ABOVE` correctly). But `feedback_amendments` — the actual steering (prompt amendments appended to the next segment's middle-layer text) — has branches for four. A maximally out-of-band palette (`0.99` vs the `0.30` ceiling the director context itself prints) and a maximal scene-boundary spike (`0.99`) yield ZERO amendments:

```
feedback_amendments({'palette_distance': 0.99, 'scene_boundary_strength': 0.99}, style) == []
feedback_amendments({'motion_energy': 0.99}, style) == ['calm static composition, minimal motion']
```

The structural reason is visible in `StyleSpec` (`models.py:67-82`): the charter defines bands for the four steered metrics and none for the other two, so the pure function cannot threshold them without inventing a band. The result is a measure-without-steering half-loop: the inspector flags palette blowout, the director reads it, and no corrective text ever reaches the next prompt.

## Rationale

- The Phase 5 contract is "each amendment nudges the next segment's text back toward the charter band" (`prompts.py:82-88`). Two measured deviations have no nudge — the loop is open for exactly the failure modes users report (highlight blowout, cut instability).
- The director-context side already decided these metrics matter (hardcoded 0.30 bands at `director.py:234,236`); the feedback side silently disagrees by omission.
- Fix cost is a charter decision (add `StyleSpec` bands vs document informational-only), not just code — filing so the decision is explicit instead of accidental.

## Live evidence (container, 2026-09-30)

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.prompts import feedback_amendments
from voyage.models import StyleSpec
style = StyleSpec(prompt='x', motion_energy_min=0.0, motion_energy_max=0.35,
    visual_complexity_max=1.0, semantic_drift_min=0.0, style_similarity_min=0.0)
print(feedback_amendments({'palette_distance': 0.99, 'scene_boundary_strength': 0.99}, style))
print(feedback_amendments({'motion_energy': 0.99}, style))"

[]
['calm static composition, minimal motion']
```

`StyleSpec` fields live: `['prompt', 'motion_energy_min', 'motion_energy_max', 'visual_complexity_min', 'visual_complexity_max', 'semantic_drift_min', 'semantic_drift_max', 'style_similarity_min', 'surrealism', 'transition_smoothness']` — no palette/scene keys.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.prompts import feedback_amendments
from voyage.models import StyleSpec
s = StyleSpec(prompt='x')
print(feedback_amendments({'palette_distance': 0.5, 'scene_boundary_strength': 0.5}, s))"
# Expect [] today (no branch exists); motion_energy=0.5 with default max 0.35 gives an amendment.
```

## Fix candidates

1. Charter decision first: add `palette_distance_max` + `scene_boundary_strength_max` to `StyleSpec` (default 0.30, matching the director-context bands) with `bands_ordered` coverage, then add the two amendment branches (e.g. palette → "muted restrained palette, no blown highlights"; boundary → "single continuous shot, no cuts").
2. Or document the two as informational-only (director sees, never steers) in the `feedback_amendments` docstring + DESIGN §43 — turns a silent gap into an explicit contract.
3. Test: each of the six metrics out-of-band in isolation → exactly the expected amendment (or the documented no-steer).

## Refs

- `voyage/prompts.py:82-107`; `voyage/director.py:230-237` (renders all six); `voyage/models.py:67-82` (charter covers four); `voyage/scoreboard.py:25-32` (six-metric contract).
- Not-a-duplicate: 144 (NaN/±inf finiteness in the same two functions — coverage, not finiteness); 026 (staged-plan truncation + blocklist — different function); 027/062 (scoreboard crash/paths — read-only view, not steering).

## Progress log

- 2026-09-30 (Group B): re-verified live first (`voyage:latest`, CPU-only):
  `feedback_amendments({'palette_distance': 0.99,
  'scene_boundary_strength': 0.99}, style)` → `[]` while motion 0.99 amends;
  `StyleSpec` fields carry no palette/scene keys. Premise confirmed.
- Charter decision (candidate 1, steering over informational-only): both
  metrics steer — palette blowout / cut instability are exactly the
  user-reported failure modes the Phase 5 loop exists to correct.
- TDD: wrote `tests/test_feedback_six_metrics_180.py` first — 4 failed / 2
  passed (in-band + non-finite pins green) before the fix.
- Fix: `StyleSpec.palette_distance_max` + `scene_boundary_strength_max`
  (default 0.30, matching the director-context hardcoded ceilings, covered
  by 119's `values_in_domain` [0,1] validator) + two `feedback_amendments`
  branches with the file's established finite-guard idiom (issue 144).
- Gates (in-container): new tests + `test_models_ranges_119` +
  `test_feedback` + `test_inspector_wiring` + `test_stage_a_telemetry` +
  `test_phase3` = 50 passed; `ruff check` + `ruff format --check` + `mypy`
  on `voyage/models.py` + `voyage/prompts.py` clean.

## Resolution

- Verdict: FIXED. Files changed: `voyage/models.py` (2 charter fields +
  validator coverage), `voyage/prompts.py` (2 amendment branches),
  `tests/test_feedback_six_metrics_180.py` (new: isolation pins for all
  six, in-band/none, non-finite/none, configurable bands + range pins).
- DESIGN proposal (quoted text only, for the DESIGN owner): in §43, after
  the amendment contract, add: "All six MEASURED metrics steer: each
  out-of-band metric appends exactly one corrective amendment to the next
  segment's middle-layer text (palette blowout → palette restraint,
  boundary spike → single-shot continuity), thresholded against the
  charter's bands."
- Residuals (other files, not touched per scope): `voyage/director.py:235`
  + `:237` still hardcode `(0.0, 0.30)` for palette/scene instead of
  reading `style.palette_distance_max` /
  `style.scene_boundary_strength_max` — wire the MEASURED-band table to the
  new charter fields so a retuned charter and the director flags cannot
  disagree.
