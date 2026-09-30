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

## Progress log (2026-09-30, resolution pass)

- Premises spot-verified live (read-only): `LongLiveBackend` /
  `AceStepBackend` still `NotImplementedError` placeholders in
  `voyage/fake_backends.py:186-206` with zero importers/callers outside
  that file; `_sha256` still present in `voyage/model_registry.py:370`.
  The issue's findings hold.
- No code change made: every fix candidate deletes or edits files under
  `voyage/` (`fake_backends`, `model_registry`, `tui`, `tui_state`,
  `cli`, `backends`, `__init__`s, `bench.py`), which is outside this
  pass's scope (`tests/` + `docs/TASK.md` only). The test-side pins
  (`tests/test_hashing.py:47`, `tests/test_tui_state.py:50`) cannot move
  first — they pin behavior the source still ships.

## Resolution

- Not resolved here — returned as residual for a `voyage/`-scoped pass.
  Recommended order per the issue (highest safety first): items 1+2+5+6
  (pure deletes/inlines), then 3+4 (shim removal), then the `__init__` /
  `bench.py` disposition. Gate each with `gates.sh` green.

## Progress log (2026-09-30, Group D pass)

- All 7 premises re-verified live (host reads, tree as-read): (1)
  `LongLiveBackend`/`AceStepBackend` still `NotImplementedError`
  placeholders at `voyage/fake_backends.py:186/200` with zero importers
  outside that file — holds. (2) `_sha256` still at
  `voyage/model_registry.py:351` (`__all__`) / `:383` (def), pinned by
  `tests/test_hashing.py:47` — holds. (3) `getattr` shims still at
  `voyage/tui.py:165,186,198` — holds, and `tui.py` gained concurrent
  uncommitted edits mid-pass (checkbox additions), so it is doubly out of
  scope. (4) `LAST_SETTINGS_PATH` still at `voyage/tui_state.py:66` +
  docstring `:9` + pin `tests/test_tui_state.py:50` — holds, same
  concurrent-edit collision. (5) `_check_run_id` cites drifted under the
  cli split (now an import at `voyage/cli.py:48` + `__all__` string at
  `:172`; the `:118-122/:1249` anchors are stale) — `cli.py` not owned.
  (6) **Refuted**: the only `1e-9` left in `voyage/` is the constant def
  itself (`backends.py:94`) plus its docstring mention (`:108`) — no
  re-literal anywhere; the cli split already removed the duplication. (7)
  `voyage/workers/__init__.py` is docstring-only (no code list — the
  "stale list" premise is a doc gap at most); `audio`/`vision`
  `__init__.py` are 1-line pointers as described; `bench.py` is 150L,
  disposition still open.
- No code change: (1) needs `fake_backends.py` (not owned); (2) needs a
  `model_registry.py` deletion, which exceeds this pass's alias-hunks-only
  scope on that file, and the test pins cannot move first; (3)(4) collide
  with concurrent edits; (5)(7) need their owning passes.

## Resolution (2026-09-30, Group D pass)

- Not resolved here — returned with precise per-item scoping above (one
  refutation: item 6 already fixed by the cli split).
- Files changed: none for 086. Gate evidence: n/a. DESIGN proposals: none.
  Residuals: items 1+2 (fake-backends / registry passes, with their test
  pins), 3+4 (tui passes after the concurrent feature work lands), 5
  (cli pass, cites need re-anchoring post-split), 7 (`__init__`/bench
  disposition).

## Progress log (2026-09-30, tests-only pass)

- All 7 premises re-verified live (read-only): (1)
  `LongLiveBackend`/`AceStepBackend` still `NotImplementedError`
  placeholders at `voyage/fake_backends.py:186/200` — holds. (2) `_sha256`
  still at `voyage/model_registry.py:383` (+ `__all__` `:351`), pinned by
  `tests/test_hashing.py:47` (`test_registry_alias_delegates`) — holds.
  (3) `getattr` shims still at `voyage/tui.py:165,186,198`
  (`_load_initial_state` / `_backend_gpu_warning` / `_save_last_settings`)
  — holds. (4) `LAST_SETTINGS_PATH` still at
  `voyage/tui_state.py:66` + docstring `:9`, pinned by
  `tests/test_tui_state.py:50` — holds. (5) `_check_run_id` now lives in
  `voyage/cli_paths.py:59` (post-split anchor; the issue's
  `cli.py:118-122/:1249` cites are stale but the wrapper still exists with
  2+ call sites via re-export) — holds in substance. (6) Still refuted:
  the only `1e-9` in `voyage/` is the `FLOAT_DUST_EPSILON` def itself
  (`voyage/backends.py:94`) + use (`:115`) + docstring — no cli re-literal
  remains. (7) `voyage/workers/__init__.py` still docstring-only (stale
  Phase-0 backend list in prose), `audio`/`vision` `__init__.py` still
  1-line pointers, `voyage/bench.py` now 232L (growth vs 150L as-read) —
  disposition still open.
- Tests-scope triage: ZERO items are fixable from `tests/` alone. Every
  fix candidate deletes or edits a `voyage/` file (frozen scope): (1)
  needs `fake_backends.py`; (2) needs a `model_registry.py` deletion (test
  pin `test_hashing.py:47` pins shipped behavior — cannot move first);
  (3) needs `tui.py`; (4) needs `tui_state.py` (+ docstring + 1 test move
  together); (5) needs `cli_paths.py`; (7) needs `__init__`s/`bench.py`.
  No code change made — inventing a tests-only "fix" (e.g. deleting the
  pins while the source still ships the aliases) would break gates, not
  fix dead code.
- Gate evidence: n/a (no files changed).

## Resolution (2026-09-30, tests-only pass)

- Verdict: blocked (no tests-scope item; all voyage-owned — recorded per
  item, not pretended). Files changed: none. DESIGN proposals: none.
- Residuals mapped per item with exact file:line + owner handoffs: (1)
  fake-backends pass — delete `voyage/fake_backends.py:186-206` classes
  (zero importers outside file, verified) + confirm `voyage/supervisor.py`
  dispatch untouched; (2) registry pass — delete
  `voyage/model_registry.py:383` `_sha256` + `:351` `__all__` entry AND
  update `tests/test_hashing.py:47` in the same commit (pin cannot move
  first); (3) tui pass (after concurrent feature work lands) — replace
  `voyage/tui.py:165,186,198` `getattr` shims with direct imports/calls +
  `importlib.util.find_spec` probe; (4) tui-state pass — delete
  `voyage/tui_state.py:66` constant, fix docstring `:9`, update
  `tests/test_tui_state.py:15,50` in the same commit; (5) cli-paths pass —
   inline `voyage/cli_paths.py:59` `_check_run_id` at its call sites
   (`cli_run_ops.py`, `cli_generate.py` via re-export) with re-anchored
   cites; (6) already fixed — no action; (7) `__init__`/bench disposition
   pass — real exports or delete pointers (`voyage/audio/__init__.py`,
   `voyage/vision/__init__.py`), refresh `voyage/workers/__init__.py`
   prose, fold-or-keep decision for 232L `voyage/bench.py`.

## Progress log (2026-09-30, voyage-scope resolution pass)

- Premises re-verified live in-container
  (`docker run --rm -v $PWD/Voyage:/app -w /app voyage:latest`,
  CPU-only, no host pip): (1) `LongLiveBackend`/`AceStepBackend` still
  `NotImplementedError` placeholders at `voyage/fake_backends.py:186/200`
  with zero importers outside that file (`grep -rn` hits only the defs +
  `.pyc`) — holds. (2) `_sha256` still at
  `voyage/model_registry.py:383` + `__all__` `:351`, sole pin
  `tests/test_hashing.py:47` — holds. (3) `getattr` shims still at
  `voyage/tui.py:181,202,214` + `import textual/del textual` probe
  `:163-174`; `tui_state` always defines `load_last_settings` (`:562`),
  `gpu_warning` (`:607`), `save_last_settings` (`:505`) — holds. (4)
  `LAST_SETTINGS_PATH` still at `voyage/tui_state.py:66` + docstring `:9`,
  pinned by `tests/test_tui_state.py:15,50`; runtime uses
  `_default_settings_path()` (`:514,572`) — holds. (5) `_check_run_id`
  still at `voyage/cli_paths.py:59` with live call sites
  (`cli_generate.py:99`, `cli_run_ops.py:42` via re-export) — holds in
  substance (NOT zero-caller). (6) Still refuted: the only `1e-9` in
  `voyage/` is the `FLOAT_DUST_EPSILON` def (`backends.py:94`) + use
  (`:115`); `cli_planning.py:16,111` imports the single source — no
  cli re-literal remains. (7) `workers/__init__.py` still docstring-only
  with stale Phase-0 prose; `audio`/`vision` `__init__.py` still 1-line
  pointers; `bench.py` 232L with live importers
  (`cli_observe.py`, `test_benchmark.py`, `test_unit.py`, …) — holds.
- Actions: (1) DELETED both placeholder classes from
  `voyage/fake_backends.py` (provably zero-caller). (2) DELETED
  `model_registry._sha256` + `__all__` entry AND removed its pin
  `tests/test_hashing.py:test_registry_alias_delegates` in the same edit
  (pin cannot move first). (3) REPLACED all three `tui.py` `getattr`
  shims with direct imports/calls (`load_last_settings`,
  `gpu_warning`, `save_last_settings` added to the `tui_state` import;
  `tui_state_module` import removed) + replaced the
  `import textual/del textual` probe with
  `importlib.util.find_spec("textual") is None` (same idiom as
  `tui_state.textual_available()`); docstring getattr line updated.
  (4) DELETED `LAST_SETTINGS_PATH`, fixed module docstring + comment to
  reference `_default_settings_path()`, updated
  `tests/test_tui_state.py` import + `test_last_settings_path_default`
  to assert the function. (5) KEPT `_check_run_id` — live helper with
  2 call sites, not dead code; inlining would churn two verbs for zero
  safety payoff. `cli_paths.py` also carries concurrent uncommitted
  edits (another track) — doubly out of scope to reshape. (6) NO ACTION
  (already fixed). (7) REFRESHED `workers/__init__.py` prose to list
  the real workers (video/video_ltxv/video_longlive/video_causvid,
  audio/audio_acestep, sfx/sfx_mmaudio, director, augment_worker);
  KEPT `audio`/`vision` 1-line pointers (deleting the docstring leaves
  an empty file — worse) and KEPT `bench.py` as the math lib (live
  importers — fold decision belongs to its owning pass).
- Gate evidence (in-container `voyage:latest`): `ruff check` clean on
  all 7 touched files; `ruff format --check` clean on all 7;
  `mypy voyage/tui.py` clean; `mypy` on the other touched files clean
  except 3 pre-existing `comparison-overlap` errors in
  `tests/test_tui_state.py:146,151,199` (verified present on HEAD via
  stash — plan-counts tests, untouched by this pass). Suites:
  `test_hashing.py + test_tui_state.py` 64 passed,
  `test_tui.py` 20 passed.

## Resolution (2026-09-30, voyage-scope resolution pass)

- Verdict: resolved except explicitly kept items below. Files changed:
  `voyage/fake_backends.py` (delete 2 placeholder classes),
  `voyage/model_registry.py` (delete `_sha256` + `__all__` entry),
  `tests/test_hashing.py` (drop registry-alias pin + import),
  `voyage/tui.py` (direct imports/calls + `find_spec` probe),
  `voyage/tui_state.py` (delete `LAST_SETTINGS_PATH`, docstring fix),
  `tests/test_tui_state.py` (import + path test → function),
  `voyage/workers/__init__.py` (prose refresh).
- DESIGN proposals: none — "No DESIGN change required: this pass deletes
  zero-caller placeholders/aliases/shims and refreshes worker-index prose;
  no behavior, contract, or geometry changes."
- Residuals: (5) `_check_run_id` thin-wrapper inline (cli-paths pass,
  after concurrent edits land); (6) none; (7) `audio`/`vision`
  `__init__` real-exports-or-delete + `bench.py` fold-or-keep (owning
  passes); `tests/test_tui_app.py:193,483` defensive `getattr` mirrors
  (test code, still passing — tui pass may direct-call them).
