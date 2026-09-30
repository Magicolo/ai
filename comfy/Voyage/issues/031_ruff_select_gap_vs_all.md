# 031 — Ruff `select` gap vs Zoomy `ALL`: thousands of findings invisible by config

- Severity: LOW (lint scope — green gate is a scope artifact, not a runtime defect)
- Group: standards/lint — Rank: 4/5
- File:line: `Voyage/pyproject.toml:70`
- Overlaps: 032/033/034 cluster (lint/typing scope debt) — same ratchet, not duplicates; fix as one toolchain pass.

## Description

Voyage selects 16 ruff families while Zoomy enforces `ALL` (line-length 100).
Probing Voyage with `--select ALL` yields thousands of un-gated hits, so the
green gate is a scope artifact, not cleanliness. Entire debt classes
(docstrings, annotations, magic values, security `S`, pytest style `PT`,
naming `N`, performance `PERF`, `PLR0913/0917` complexity) stay permanently
dark.

## Rationale

Ruff docs recommend starting small and adding a category at a time with
explicit `select` — but the project norm (AGENTS.md §12) is `ALL` parity with
Zoomy. Without a ratchet plan the current select freezes the gap: new code in
unselected families lands ungated and reviewers assume coverage that does
not exist.

## Live evidence (re-verified 2026-09-30)

`Voyage/pyproject.toml:70` (read live):

```toml
select = ["E", "F", "I", "UP", "B", "A", "C4", "DTZ", "W", "BLE", "TRY", "EM", "SIM", "RUF100", "S101", "T201"]
```

Track-C sweep capture (in-container `ruff check --select ALL --statistics`,
quoted from preserved Task output `ses_f10013fc5ffeLLDtqZEwFbf3JR`):

```
D103 771, COM812 511, TRY003 377, PLR2004 328, SLF001 290, EM102 259,
ANN401 255, CPY001 137, EM101 124, TC003 79, D102 76, ARG001 69,
PLR0913 52, PLR0917 41, FBT001 39, PT011 38 …
```

Gate itself is green on the scoped select (`ruff check .`: All checks passed).

## Repro

From `Voyage/` (needs the gate image):

```bash
docker run --rm -v "$PWD:/app" voyage:latest bash -c "ruff check --select ALL --statistics . 2>&1 | head -n 60"
```

Compare `Voyage/pyproject.toml:70` vs `Zoomy/pyproject.toml` (`select = ["ALL"]`).

## Fix candidates

(a) Ratchet `select` upward family-by-family with per-family autofix passes.
(b) Declare the Zoomy-parity gap explicitly in `pyproject.toml` comment +
`TASK.md` with an ordered adoption list.
(c) At minimum enable `ANN,D,PLR2004,PT,S,PERF,N` in CI as warn-only before
enforcing.

## Refs

- `Voyage/pyproject.toml:70`
- Ruff linter docs: "Use ALL with discretion…" / "Prefer lint.select… Start
  with a small set and add a group at-a-time" (docs.astral.sh/ruff/linter).
- AGENTS.md §12 (Zoomy-parity toolchain posture).
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §1.

## Progress log (2026-09-30, toolchain track)

- Re-verified premises live in-container (`voyage:latest`, CPU-only):
  `ruff check --select ALL --statistics .` now reports (as-read 2026-09-30):
  D103 797, COM812 703, PLC0415 576, TRY003 461, SLF001 457, PLR2004 387,
  ANN401 326, EM102 324, CPY001 167, EM101 143, ARG001 110, TC003 95 —
  every family grew vs the preserved sweep (D103 771→797, COM812 511→703,
  TRY003 377→461, PLR2004 328→387, ANN401 255→326), so counts drifted up
  with tree growth. `pyproject.toml:70` select is unchanged (16 families).
- Probed each ordered-adoption candidate family in-container
  (`ruff check --select FAMILY --statistics .`):
  ANN 339 (ANN401 326 + ANN001 10 + ANN202 3), D ~900+ (D103 797, D102 73,
  D205 35, …), PLR2004 387, PT 85 (PT011 42 + PT018 41 + …), S 77 (S603 31
  + S607 19 + S108 18 + S110 7 + S112 2), PERF 10 (PERF401 5 + PERF203 5),
  N 39 (N806 34 + N802 4 + N818 1). **No candidate family is green** —
  not even the narrowest (PERF: 10 hits across 7 files incl.
  tests/test_tui_app.py, voyage/cli_observe.py, voyage/cli_validate.py,
  voyage/concepts.py, voyage/models_ensure.py, voyage/supervisor.py ×2,
  voyage/workers/director.py ×2, voyage/workers/video_longlive.py;
  fixing them is loop restructuring across dirty/foreign files, out of
  this track's scope).
- Verdict per family (all stay out this pass): ANN — 339 hits, `Any` is
  the default seam type (see 035); D — ~900 hits, undocumented-public
  sweep needs its own pass; PLR2004 — 387 tree-wide, only the 15
  top-level hits convert this pass (see 038; scoped tripwire added to
  gates.sh instead of select); PT — 85 hits, test-style pass belongs to
  the test-structure track (039/040/086/088/089, explicitly out of
  scope); S — 77 hits, subprocess/tempfile hardening is behavior-adjacent
  (needs per-site review, not autofix); PERF — 10 hits but scattered
  across dirty/foreign files; N — 39 hits, naming pass belongs with the
  relevant owners.
- `select` left unchanged (no family added). No pyproject change for 031.

## Resolution

- Document-only: the gap stays policy (per-file ratchet continues via
  032/033/034/038/094), with fresh as-read counts above replacing the
  preserved sweep numbers. Next pass should retry PERF first (smallest:
  10) once the owning files settle, then N (39), then PT with the test
  track.
- Files changed: none for 031. Gate evidence: `select` untouched, so
  `ruff check .` scope is identical (its 7 current errors are all
  foreign in-flight files — see 032 log). DESIGN proposals: none.
  Residuals: entire adoption list (ANN, D, PLR2004-full, PT, S, PERF, N)
  still dark; counts re-baselined above.

## Progress log (2026-09-30, Group D pass)

- Re-verified live in-container (`voyage:latest`): `ruff check --select PERF .`
  → **11 hits** (was 10): `tests/test_tui_app.py:720` (PERF401),
  `voyage/cli_observe.py:352` (PERF203), `voyage/cli_validate.py:223,229`
  (PERF401), `voyage/concepts.py:431` (PERF203),
  `voyage/models_ensure.py:222` (PERF401), `voyage/supervisor.py:591,663`,
  `voyage/workers/director.py:390,517`, `voyage/workers/video_longlive.py:348`.
  `ruff check --select N --statistics .` → **39** (N806 34 + N802 4 + N818 1,
  unchanged shape). `ruff check --select PT --statistics .` → **85**
  (PT011 42 + PT018 41 + PT013 1 + PT012 1, unchanged shape).
- Retry order PERF → N → PT (batch-7 prescription): PERF is red (11 hits
  across cli_observe/cli_validate/concepts/models_ensure/dirty-supervisor/
  workers — loop restructuring across dirty + foreign files, out of scope),
  so N and PT were probed for counts only and the adoption stops: **no
  family is green in isolation, no `select` change**.
- `pyproject.toml:70` select confirmed unchanged (16 families).

## Resolution (2026-09-30, Group D pass)

- Document-only: gap stays policy; as-read counts re-baselined above
  (PERF 10→11, N 39, PT 85). Next pass retries PERF first once the owning
  files settle.
- Files changed: none for 031. Gate evidence: `select` untouched.
  DESIGN proposals: none. Residuals: full adoption list (ANN, D,
  PLR2004-full, PT, S, PERF, N) still dark.
