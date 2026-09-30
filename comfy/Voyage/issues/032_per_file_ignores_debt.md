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

## Progress log (2026-09-30, toolchain track)

- Re-verified each `per-file-ignores` entry live in-container by running
  its ignored rule set against the file with `--isolated` (config
  ignores off), as-read 2026-09-30 — top-level `voyage/*.py`:
  tui.py 25, supervisor.py 13, cli.py 226, doctor.py 6, config.py 1,
  logrotate.py 6, concepts.py 2, prompts.py 1, tui_state.py 1,
  model_registry.py 1, persistence.py 1 (RUF100), media.py 1 (T201),
  console.py **0** (T201), models_ensure.py 1 (BLE). Workers: loop.py 3,
  director.py 6, video_causvid.py 13, video_longlive.py 12,
  video_ltxv.py 11, audio_acestep.py 1. Tests: observability 1 (RUF100),
  backend_registry/causvid_prep/doctor/tui_state 1 each (SIM300),
  commit_hardening 3 (SIM105), console 1 (SIM117), tui_app 1 (SIM108).
  (One probe round ran from the wrong mount root and returned all-zero —
  discarded; re-ran with the correct `-v $PWD:/app -w /app`.)
- Only one entry is green: `voyage/console.py` T201. Root cause is a
  ruff-version behavior, verified by probe: ruff 0.16.9 T201 exempts
  `print(..., file=...)`, and all three prints in console.py
  (`:131,143,155`) pass `file=self._stream` — a bare-`print` probe file
  flags line 1 only, confirming the exemption. The entry is stale.
- Converted this pass (delete the entry, not the rule): removed
  `"voyage/console.py" = ["T201"]` from `pyproject.toml`
  per-file-ignores. Verified under the real gate config in-container:
  `ruff check voyage/console.py` → All checks passed (entry genuinely
  unneeded, not masked elsewhere).
- Every other entry still fires (≥1 hit), so no further conversion:
  removing any of them would redden `ruff check .`. Closest runners-up
  (1 hit each: config TRY004, prompts SIM108, tui_state SIM103,
  model_registry S101, persistence RUF100, media T201, models_ensure BLE,
  audio_acestep TRY004, most test SIM/RUF100 entries) each need a code
  fix in a file outside this track's scope (workers/tests/dirty files),
  so they stay with their owning passes.
- TDD/regression: the removal itself is the tripwire — `ruff check .`
  in gates.sh now fails if a bare `print()` ever lands in console.py
  (previously silently exempted).

## Resolution

- One conversion: `voyage/console.py` off T201 (pyproject.toml only, no
  code change — zero behavior). Gate evidence: in-container
  `ruff check voyage/console.py` green post-removal; full
  `ruff check .` shows 7 errors, all in foreign in-flight files
  (voyage/media.py F401 from the concurrent augment-import change;
  tests/test_cli_split.py, tests/test_registry_split.py,
  tests/test_single_source.py I001/SIM300 from the concurrent cli-split
  migration) — none in this track's scope, none caused by this removal.
- Files changed: `pyproject.toml` (one line deleted). DESIGN proposals:
  none. Residuals: all remaining per-file-ignores entries (still-firing
  counts recorded above); RUF100-masked files still cannot self-clean —
  unmasking needs the owning code passes.
