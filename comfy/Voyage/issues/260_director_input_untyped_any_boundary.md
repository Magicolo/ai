# 260 — `director_input_from_state(state: Any, …)` — untyped boundary crossing under `mypy strict`

Severity: MEDIUM (track C-02).

## Technical description

The §20 bounded-input assembler takes `state: Any` (`voyage/director.py:289`), then
touches `state.current_concept`, `state.destination_concept`, `state.phase`,
`state.decision_index`, `state.committed_segments`, `state.timeline_frames` — all
unchecked. One typo'd attribute becomes a runtime `AttributeError` that strict mypy was
supposed to preclude.

## Rationale

mypy docs: "`Any` is compatible with every type" and disables checking at and downstream
of the annotation. A boundary constructor is exactly where a precise type pays off; the
supervisor state type exists in-tree (`RunState`/persistence models). The 9 `ANN401` hits
in scope cluster here and in `console.py` task handles.

## Live evidence

```
$ ruff check --select ANN,D,... (dark probe): 9× ANN401
voyage/director.py:289:12: ANN401 Dynamically typed expressions (typing.Any) are disallowed in `state`
voyage/console.py:703:35, :726:58, :736:33, :741:36, :746:35 — ANN401 on task/progress handles
voyage/boundary_metrics.py:177:21, :182:26, :189:20 — ANN401 on _as_text* helpers
```

(Note: host has no ruff by design — dark probes ran in-container `voyage:latest`.)

Repro: `docker run --rm -v "$PWD:/app" -w /app voyage:latest ruff check --select ANN
voyage/director.py` → ANN401 at 289:12. Or rename one attribute in a scratch copy — mypy
stays silent.

## Source refs

`voyage/director.py:288-314`.

## Online sources

- mypy docs on `Any` ("no type checking … assign it to any variable", python.org typing
  docs); mypy `strict` contract in `pyproject.toml:139`.

## Fix candidates

- (a) type `state` as the real supervisor state (import-time cost? `director.py` already
  imports `voyage.models` — check for a cycle first; else a `Protocol` with the six
  attributes); (b) adopt `ANN401` for this file only (remove from the dark list per-file
  as the owning pass converts it, per pyproject retry-order note).

## Log

- 2026-10-07: filed from read-only Track C sweep; no code touched.

## Evaluation (2026-10-07)
Live probe in-container: `director_input_from_state` signature is
`(state: 'Any', ...)` (stringified by `from __future__ import
annotations`); it touches six attributes (`current_concept`,
`destination_concept`, `phase`, `decision_index`, `committed_segments`,
`timeline_frames`) unchecked. Cycle check: `director.py` already imports
`voyage.models` and `models.py` never imports `director`, so the real
`RunState` import is free — no `Protocol` needed. `RunState` carries all
six fields. Confirmed as filed; fix is option (a) with the real type.

## Progress log
- `director.py`: imported `RunState` from `voyage.models`; typed
  `state: RunState` with a docstring naming the six attributes + no-cycle
  rationale (kept `Protocol` import for the existing `DirectorBackend`).
- New tests in `tests/test_issue_260_director_state.py` (2 tests):
  `get_type_hints` pins `state is RunState`; six-attribute flow pin
  through a real `RunState` (world/transition/metrics strings).
- Verified: `ruff check` + `format --check` clean; `mypy` strict clean
  on `director.py`; scoped pytest 113 passed.

## Resolution (2026-10-07)
Fixed as proposed via the real type (no `Protocol` fallback needed).
No open items; `ANN401` adoption for the remaining dark-list sites
(`console.py`, `boundary_metrics.py`) stays with the owning passes.
