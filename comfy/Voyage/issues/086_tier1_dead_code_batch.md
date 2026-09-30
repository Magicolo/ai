# 086 — Tier-1 dead code batch: stubs, shims, aliases, constants (zero-caller deletes)

- Severity: MEDIUM (dead code — safe first)
- Area: hygiene — outright dead / superseded
- Decision: Phase 0 unblocker; Q&A aggressive = land this first

## Description

Zero-caller or shim-only sites verified live 2026-09-30:

1. `voyage/fake_backends.py:186-206` — `LongLiveBackend`/`AceStepBackend` `NotImplementedError` Phase-0 placeholders ("lands in Phase 1/2 (E)", "lands in Phase 4 (H)"). Dispatch is `supervisor.py:88-108`, never these classes; nothing imports them.
2. `voyage/model_registry.py:312-317` — `_sha256()` legacy alias of `hashing.sha256_file` (issue 021). Prod uses `sha256_file` directly; sole pin `tests/test_hashing.py:47`.
3. `voyage/tui.py:161-202` — `getattr` shims for "Stream A may not exist yet" (`load_last_settings`/`gpu_warning`/`save_last_settings` + `tui.py:145-156` `import textual/noqa F401; del textual` probe). Stream A landed; `tui_state.py:440,495,541` always defines them.
4. `voyage/tui_state.py:45` — `LAST_SETTINGS_PATH` import-time `Path.home()` snapshot vs `:48-55` `_default_settings_path()` call-time resolution (comment admits requirement). Runtime uses the function; constant used only by docstring `:9` + `tests/test_tui_state.py:50`.
5. `voyage/cli.py:118-122` — `_check_run_id()` thin wrapper over `is_flat_folder_name` (`:101`), 2 call sites (`:172,1249`).
6. `voyage/backends.py:94` vs `voyage/cli.py:1123` — `FLOAT_DUST_EPSILON = 1e-9` defined once, cli re-literals `1e-9` (docstring at `backends.py:108` cites `cli.segments_for_duration`).
7. `voyage/__init__.py` (7L, version only — fine, no change); `voyage/audio/__init__.py` + `voyage/vision/__init__.py` (1L pointer stubs, no exports); `voyage/bench.py` (57L, 2 pure fns — fold into console/scoreboard or keep as math lib); `voyage/workers/__init__.py:6-11` stale Phase-0 backend list (omits ltxv/causvid/sfx/augment).

## Rationale

Highest safety×payoff: pure confusion surface, no behavior change, unblocks god-module splits (fewer names to move).

## Live evidence

- `rg -n "LongLiveBackend|AceStepBackend|_sha256\(|_check_run_id|LAST_SETTINGS_PATH|FLOAT_DUST_EPSILON|1e-9" voyage/ tests/`
- `cat voyage/audio/__init__.py voyage/vision/__init__.py voyage/workers/__init__.py`

## Repro

```bash
grep -rn "LongLiveBackend\|AceStepBackend" voyage/ tests/ | head
grep -n "_sha256\|getattr(tui_state" voyage/model_registry.py voyage/tui.py tests/test_hashing.py
```

## Fix candidates

1. Delete 1+2+5+6 (classes, alias+1 test line, wrapper inline, constant import); replace 3 with direct imports/calls + `importlib.util.find_spec` probe (same idiom as `cli.py:1129`, `tui_state.py:118`); delete 4 constant, fix docstring+1 test.
2. Give `__init__`s real exports or delete pointers; refresh worker index doc; fold or keep `bench.py` explicitly.
3. Gate: `gates.sh` green.

## Refs

- `voyage/hashing.py:30` (canonical), `voyage/tui_state.py:440,495,541`, `voyage/cli.py:99,1129`
