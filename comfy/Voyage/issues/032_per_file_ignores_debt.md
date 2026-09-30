# 032 — 20+ `per-file-ignores` entries institutionalize debt (incl. self-policing `RUF100`)

- Severity: LOW (lint hygiene — permanent exemptions without owner/date)
- Group: standards/lint — Rank: 4/5
- File:line: `Voyage/pyproject.toml:80-113` (whole block)
- Overlaps: 031/033/034 cluster — same ratchet, not duplicates; never add new `RUF100` ignores.

## Description

18 source + 8 test per-file exemptions suppress `BLE` (blind-except),
`SIM105/108/103/117/300`, `TRY004/300/301`, `S101`, `T201`, and critically
`RUF100` (unused-`noqa` detector) in 9 files. Ignoring `RUF100` blinds the
repo to stale suppressions — the exact failure mode already observed with
stale `type: ignore`s (issue 034 in this range).

## Rationale

Per-file ignores are the sanctioned migration tool, but each entry without a
tracked owner/date becomes permanent. `RUF100`-ignored files cannot
self-clean; `BLE`-ignored workers hide bare-`except` growth; `S101`-ignored
shipped modules (`supervisor, cli, model_registry, video_*`) ship asserts as
runtime checks (asserts vanish under `python -O`).

## Live evidence (re-verified 2026-09-30)

`Voyage/pyproject.toml:80-113` read live — key lines:

```toml
"voyage/tui.py" = ["BLE", "SIM105", "SIM117", "T201", "RUF100"]        # :86
"voyage/supervisor.py" = ["BLE", "SIM105", "S101", "RUF100"]           # :87
"voyage/cli.py" = ["BLE", "SIM105", "S101", "T201"]                    # :88
"voyage/workers/video_ltxv.py" = ["TRY004", "TRY301", "S101", "T201", "RUF100"]  # :104
```

Full block spans `:80-113` (tests `S101`, scripts `T201`, plus per-file
034 follow-ups). Live `noqa` inventory in the sweep is only ~15 lines, so
`RUF100` coverage matters disproportionately.

## Repro

```bash
rg -n "per-file-ignores" -A 35 pyproject.toml
ruff check --select RUF100 .   # compare vs per-file masked result
```

## Fix candidates

Convert one file per pass (delete the entry, not the rule — the file's own
comment at `pyproject.toml:85` already prescribes this); never add new
`RUF100` ignores; add a `warn_unused_ignores`-style CI check for `noqa`
staleness where `RUF100` is masked.

## Refs

- `Voyage/pyproject.toml:80-113`
- Ruff per-file-ignores docs (`__init__.py = ["E402"]` pattern).
- Mypy `warn_unused_ignores` rationale (config_file.rst).
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §2.
