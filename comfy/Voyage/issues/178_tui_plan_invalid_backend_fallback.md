# 178 — TUI plan shows a confident fallback plan for an invalid backend instead of "cannot plan"

- **Severity:** LOW-MEDIUM (planning display lies while the errors line tells the truth; same input, two answers)
- **Track:** A (TUI plan math tails — below 143-149/111/024)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; concurrent uncommitted edits in `voyage/tui_state.py` — lines as-read)

## File:line (live-verified)

- `voyage/tui_state.py:329-332` (`_planning_frames_and_fps`: `if backend not in BACKENDS: return (_FALLBACK_FRAMES_PER_SEGMENT, _FALLBACK_FPS)` — 48f @ 24fps)
- `voyage/tui_state.py:358-382` (`_plan_details`: parses duration + blocks, then calls `_planning_frames_and_fps(state.backend, blocks)` with NO backend-validity gate)
- `voyage/tui_state.py:394-416` (`plan_summary`: re-parses duration + blocks, never checks `state.backend in BACKENDS`, delegates to `_plan_details`)
- `voyage/tui_state.py:175-176` (`field_errors`: `if state.backend not in BACKENDS: errors["backend"] = ...` — the errors line correctly flags it)
- `voyage/tui_state.py:33-41` (`BACKENDS` tuple + `_FALLBACK_*` constants)

## Description

`field_errors` rejects an unknown backend (`backend must be one of ltxv, longlive2, causvid, fake`), so Generate is correctly blocked. But the plan line — computed by a different function that never checks the backend — happily renders a fallback plan built from the 48f @ 24fps constants: `≈6.0s · 3 segment(s) · 144 frames · bogus 48f/segment @ 24fps`. The form simultaneously says "backend is invalid" (errors line) and "here is your 3-segment plan" (plan line). The fallback exists for genuinely-unknown future backends inside `_planning_frames_and_fps`, but `_plan_details`/`plan_summary` call it with user-typed form input where "unknown" means "typo", not "future backend".

## Rationale

- Two views of the same input must agree: the errors line blocks Generate while the plan line quotes segment counts the run will never produce (the CLI single source `cli._frames_per_segment` has no such fallback — it returns `config.video.segment_frames`, and an invalid backend never reaches it because config validation rejects the Literal first).
- The fallback masks typos: `ltxx` (one transposed letter) plans at fake geometry instead of saying "cannot plan: bad backend", so the user fixes the wrong field first.
- Cheap fix (one gate), real confusion avoided on every typo.

## Live evidence (container, 2026-09-30)

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.tui_state import GenerateFormState, _plan_details, plan_summary, plan_counts
s = GenerateFormState(style='x', name='t', backend='bogus')
print('plan_details bogus:', _plan_details(s))
print('plan_summary bogus:', plan_summary(s))
print('plan_counts bogus:', plan_counts(s))"

plan_details bogus: (3, 144, 6.0, 48, 24)
plan_summary bogus: ≈6.0s · 3 segment(s) · 144 frames · bogus 48f/segment @ 24fps
plan_counts bogus: (3, 144, 6.0)
```

Contrast: `field_errors(GenerateFormState(style='x', name='t', backend='bogus'))` → `{'backend': "backend must be one of ..."}` (correctly invalid).

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.tui_state import GenerateFormState, plan_summary, field_errors
s = GenerateFormState(style='x', name='t', backend='bogus')
print(field_errors(s))   # backend error (truth)
print(plan_summary(s))   # confident plan (lie)
"
# Expect: error dict non-empty AND plan string with 'bogus 48f/segment' — they disagree.
```

## Fix candidates

1. Gate `_plan_details` on `state.backend in BACKENDS` (return `None` when invalid) — `plan_counts` becomes `None` and `plan_summary` falls into its existing `"cannot plan: ..."` branch; add the backend message there.
2. Alternatively, thread the `field_errors` backend message into `plan_summary` so the plan line reads `cannot plan: backend must be one of ...` (single source with the errors line).
3. Test: invalid backend → `plan_counts is None` + `plan_summary` starts with `cannot plan`; valid backends unchanged.

## Refs

- `voyage/tui_state.py:329-332,358-382,394-416` vs `:175-176`; `voyage/cli.py:1158-1168` (`_frames_per_segment` — no fallback, config Literal rejects first).
- Not-a-duplicate: 024 (single-source frames/fps math — this is the invalid-input branch it never gated); 111 (take_seconds > ahead invariant — different field); 062 (finite guards — different layer).

## Progress log

- 2026-09-30 (surface-rank2 track): premise re-verified live — `GenerateFormState(backend='bogus')` gave `_plan_details == (3, 144, 6.0, 48, 24)`, `plan_summary == '≈6.0s · 3 segment(s) · 144 frames · bogus 48f/segment @ 24fps'` while `field_errors` correctly flagged the backend. Verdict: CONFIRMED. (Track brief labels this "024"; the content is this file 178 — real 024 is the unbounded-output-paths CLI issue, whose validate/finalize/sfx regions are owned by concurrent groups and were not touched.)
- Fix (tui_state only): `_plan_details` returns None for unknown backends; `plan_summary` returns `cannot plan: backend must be one of ...` (same wording as the errors line); `plan_counts` goes None through the existing struct. `_planning_frames_and_fps` fallback untouched (genuinely-unknown future backends, not form typos). Valid-backend plans byte-identical (control test).
- Evidence: 3 new 178 tests failed pre-fix, pass post-fix; tui suites green; ruff + format-check + mypy strict clean.

## Resolution

- FIXED 2026-09-30: the TUI plan line agrees with the errors line — unknown backend renders invalid/blocked, never a confident fallback plan.
