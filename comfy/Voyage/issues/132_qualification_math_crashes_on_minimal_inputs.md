# 132 — Qualification math crashes on minimal inputs: single-segment `summarize_run` + `steady_state_mean` + empty `time_to_first_output`

- **Severity:** LOW-MEDIUM (tests/harness — the §137A helper crashes on the smallest honest inputs instead of reporting; docstring promises only `FileNotFoundError`)
- **File:line:** `Voyage/tests/test_qualification.py:73-75` (`time_to_first_output`: `elapsed_seconds[0]`), `:78-81` (`steady_state_mean`: `sum(measured) / len(measured)` over `elapsed[1:]`), `:122-128` (`summarize_run` docstring: "Raises FileNotFoundError when a segment video is missing"), `:154-155` (`within_mean = sum/len`, `boundary_mean = sum/len` with no empty guard)
- **Area:** (b) `tests/test_qualification.py` gaps below previous windows

## Description

Three pure helpers in the qualification harness divide/index without guards, and the suite only ever feeds them 3-segment runs — so the crash inputs are exactly the ones a manual/smoke run produces:

1. `steady_state_mean([20.0])` → `ZeroDivisionError: division by zero` (verified live in-container, see Evidence). Any single-segment run summarized through `summarize_run` (`:161`: `steady_state_mean_s = steady_state_mean(elapsed)` with a 1-element `elapsed`) dies here before continuity is even computed.
2. `time_to_first_output([])` → `IndexError: list index out of range` (verified live). A run with zero `segment_committed` events (e.g. summarizing a fresh/failed run dir whose `metrics.jsonl` exists but has no commits) crashes instead of reporting "no segments".
3. `summarize_run` on exactly one committed segment: `boundary_diffs` is `pairwise` over one id → empty → `sum([]) / len([])` → `ZeroDivisionError` at `:155` (code-read; same shape as (1)). `within_diffs` empties the same way if `sample_frames` ever yields < 2 frames (`pairwise` over one frame), crashing at `:154`.
4. `steady_state_ratio` inherits (1) (`:84-86` divides by `steady_state_mean` — `steady_state_ratio([20.0])` → `ZeroDivisionError`, verified live), and `seconds_per_wall_second(48, 24, 0.0)` → `ZeroDivisionError: float division by zero` (verified live; a zero-wall clock reading is plausible on coarse timers).

Reachability: `scripts/qualify.sh:29` runs `--segments 3`, so the scripted leg never trips these — but the §137A procedure starts with a *smoke* leg ("one new frame"), and any operator summarizing a 1-segment smoke run (`summarize_run` is backend-agnostic by design, `:122`) hits (1)+(3). The docstring's only documented failure is `FileNotFoundError` for a missing segment video.

## Rationale (non-overlap)

- 060 (benchmark env thin) and 064 (qualify.sh gate/paths/artifacts) cover env capture, the `nvidia-smi` gate, relative paths, and stdout-only artifacts — none touches the math helpers' domains. 105 refines 064's gate mechanism (pipefail abort) — different line, different symptom.
- 051 (bench drops VRAM/percentiles) is about report content, not helper crashes.
- 126 (vision metrics edges: `estimate_frame_total` inf/nan, `select_frame_indices` duplication, `width=0` divmod) covers `voyage/vision/metrics.py` — a different module; the `pairwise`-empty shape here is in the *test harness*, not the shipped metrics.
- `test_steady_state_math` (`:200-204`) pins only the 3-element case; `test_fake_three_segment_dry_run` (`:226-242`) only the 3-segment case — no existing test feeds 0/1 inputs.

## Live evidence (verified 2026-09-30, in-container `voyage:latest`, tree as-read)

```
$ docker run --rm -v $PWD/Voyage:/app -w /app voyage:latest python3 -c "
from tests.test_qualification import steady_state_mean, time_to_first_output
..."
steady1: ZeroDivisionError division by zero
tfo-empty: IndexError list index out of range
```

- `tests/test_qualification.py:78-81`: `measured = elapsed_seconds[1:]; return sum(measured) / len(measured)` — 1-element input → `sum([])/0`.
- `tests/test_qualification.py:154-155`: `within_mean = sum(within_diffs) / len(within_diffs)` / `boundary_mean = sum(boundary_diffs) / len(boundary_diffs)` — single-segment run → `boundary_diffs == []`.
- `scripts/qualify.sh:29`: `--segments 3` — why the suite never sees it.

## Repro

```bash
docker run --rm -v $PWD/Voyage:/app -w /app voyage:latest python3 -c "
from tests.test_qualification import steady_state_mean, time_to_first_output
steady_state_mean([20.0])   # ZeroDivisionError
time_to_first_output([])    # IndexError
"
# Single-segment summarize: init a run, commit 1 segment on fake backends,
# then summarize_run(run_dir) -> ZeroDivisionError at boundary_mean.
```

## Fix candidates

1. Guard the helpers: `steady_state_mean` returns the single value (or NaN + documented) for 1-element input; `time_to_first_output` raises a typed `ValueError("no segments")` instead of `IndexError`; `summarize_run` reports `boundary_* = None` / `verdict "N/A (single segment)"` when there are no boundaries.
2. Document the new contracts in the `summarize_run` docstring next to the existing `FileNotFoundError` note.
3. Add 0/1-input tests (`test_steady_state_single_segment`, `test_summarize_single_segment_reports_no_boundary`) — the TDD pair this file's scope asks for.

## Refs

- `Voyage/tests/test_qualification.py:51-63` (REQUIRED_GPU_FIELDS), `:122-164` (`summarize_run`), `:200-242` (existing math/dry-run tests); `Voyage/scripts/qualify.sh:29`; `Voyage/issues/060_*`, `064_*`, `105_*` (adjacent harness issues, disjoint lines).

## Progress log (2026-09-30, Group D pass)

- TDD red first (in-container): 4 new tests failed as filed —
  `steady_state_mean([20.0])` ZeroDivisionError, `time_to_first_output([])`
  IndexError, `seconds_per_wall_second(48, 24, 0.0)` ZeroDivisionError,
  single-segment `summarize_run` ZeroDivisionError at `boundary_mean`.
  (Found in red phase: `steady_state_ratio([])` crashed on its own `[0]`
  read before the mean's guard — guarded at its top too.)
- Guards landed: empty input → `ValueError("no segments…")` on all three
  helpers; single-sample mean returns the sample (ratio then 1.0);
  `wall_seconds <= 0` → `ValueError`; `summarize_run` reports
  `within_mean`/`boundary_mean`/`boundary_ratio` None with verdict
  `"N/A (single segment)"` (< 2 segments) or `"N/A (no frame pairs)"`
  (degenerate sampling); docstring documents the contracts next to the
  existing `FileNotFoundError` note.

## Resolution (2026-09-30, Group D pass)

- Resolved: smoke-leg (1-segment) and zero-commit runs report instead of
  crashing.
- Files changed: `tests/test_qualification.py` (4 new tests + 4 guards +
  docstring). Gate evidence: file 14/14 in-container (incl. a 1-segment
  fake-backend summarize); `ruff check` + `ruff format --check` clean;
  ad-hoc `mypy` clean on all new hunks (5 remaining errors are
  pre-existing numpy-`Any` alias complaints on untouched lines — file is
  outside the `gates.sh` mypy list). DESIGN proposals: none (harness-only).
  Residuals: none.
