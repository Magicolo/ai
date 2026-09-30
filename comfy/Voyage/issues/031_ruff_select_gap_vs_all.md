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
