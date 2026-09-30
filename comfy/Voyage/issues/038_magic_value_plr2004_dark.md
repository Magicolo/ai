# 038 — Magic-value comparisons dark: `PLR2004` 328 hits, unselected

- Severity: LOW
- File: `Voyage/pyproject.toml:70` (absence of `PLR`); instances across `voyage/`
- Area: standards / naming
- Overlaps with: 031/032/033/034 (ruff/mypy ratchet cluster — this file is the PLR2004 slice, those are the gate/ignore slices)

## Description

`PLR2004 magic-value-comparison` fires ~328 times under ALL but `PLR` is not
selected, so raw numeric comparisons (`== 0`, `> 32`, `< 720`,
second/minute/hour multipliers, HTTP codes) accumulate without named
constants — directly against §12 ("no magic numbers, named constants with
*why*, e.g. sync-frame floor 17").

## Rationale

The codebase already does this right in places (`voyage/seeds.py:14`
`_MAX_SEED = 2**31 - 1` with a *why* comment; `tests/conftest.py:55`
`bounded_counts`, `MAXIMUM_SEED_VALUE` in property tests). Unselected
PLR2004 means new code does not follow the good examples.

## Live evidence (re-verified 2026-09-30)

`Voyage/pyproject.toml:70` read live — `select` contains no `PLR` family:

```toml
select = ["E", "F", "I", "UP", "B", "A", "C4", "DTZ", "W", "BLE", "TRY", "EM", "SIM", "RUF100", "S101", "T201"]
```

Track-C sweep `ALL` stats (preserved Task output
`ses_f10013fc5ffeLLDtqZEwFbf3JR`): `PLR2004 328`. Positive control —
`voyage/seeds.py:14` shows the required pattern with its *why* comment.

## Repro

```bash
ruff check --select PLR2004 --statistics .
ruff check --select PLR2004 voyage/ | head -n 40
```

## Fix candidates

Enable `PLR2004` alone (narrowest useful slice of `PLR`), extract constants
per module with *why* comments; keep test-only thresholds local but named.

## Refs

- Ruff `PLR2004` docs; AGENTS.md §12 naming/constants clause.
- `voyage/seeds.py:14` (good-example pattern).
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §8.
