# 080 — Split `cli.py` god module (2030L, 73 fns, 16 verbs)

- Severity: HIGH (structure)
- File: `voyage/cli.py:1` (2105 lines, 73 functions/classes per sweep)
- Area: structure — CLI decomposition
- Decision: Q&A locked — split all god modules, cli included, order at orchestrator discretion

## Description

`cli.py` mixes argparse construction, duration math, planning, disk guards, orchestration (`cmd_generate` 176L = init+run+validate+finalize), `cmd_models` 114L, `validate_run` + `finalize` + `inspect` + `benchmark/soak`, view rendering (`status` novelty/slowest-stage), CUDA preflight (`_torch_available/_cuda_stack_error/_require_cuda_stack` ~50L). Largest `cmd_generate`/`cmd_models`/`cmd_run` alone ~380L. Aliases `_run_dir_arg` + `resolve_run_dir` coexist; `_RESERVED_FOLDER_NAMES` mirrors TUI; `_check_run_id()` is a 5-line wrapper over `is_flat_folder_name` (2 call sites).

## Rationale

AGENTS.md §12 ~500-line split signal + god-module watch on `cli/supervisor`/video workers. `cli.py` is 4x the signal, highest merge-contention file with concurrent agents (§9). Every CLI addition (SFX flags, augment floors, director choices) grows the same file.

## Live evidence

- `wc -l voyage/cli.py` → 2105 (sweep said 2030 — growth, same finding)
- `rg -n "^def cmd_" voyage/cli.py` → 16× `cmd_*`
- `voyage/cli.py:118-122` `_check_run_id` wrapper; `:101` `is_flat_folder_name`; `:172,1249` call sites
- `voyage/cli.py:1171` `math.ceil(... - 1e-9)` re-literal vs `backends.py:94` `FLOAT_DUST_EPSILON`
- `voyage/cli.py:1143-1163` `_LTXV_NOVEL_BLOCK_FRAMES=96/_LONGLIVE_*` duplicating registry `segment_frames`
- `voyage/cli.py:959-986` `_augment_overrides` vs `models_ensure` weight-set seam

## Repro

```bash
wc -l voyage/cli.py; grep -n "^def \|^class " voyage/cli.py | head -n 80
grep -n "_run_dir_arg\|resolve_run_dir\|_check_run_id\|_CUDA_BACKENDS\|_frames_per_segment" voyage/cli.py
```

## Drift note (2026-09-30 pre-work re-verification, live in-container)

- `wc -l voyage/cli.py` → **2403** (was 2105 at issue filing, 2030 at sweep — growth, same finding).
- `grep -c "^def \|^class "` → **78** (was 73); `cmd_*` count is now **15**
  (init/doctor/models/run/status/pause/resume/stop/validate/finalize/sfx/
  generate/benchmark/soak/inspect) — the issue's "16 verbs" is stale by one.
- Two module-level constants missed in the original evidence also move:
  `_SEGMENT_ID_PATTERN` (validate) + `_ORPHAN_PATTERNS` (validate) +
  `_DURATION_EXAMPLES`/`_DURATION_PATTERN` (planning; the former is
  referenced by the retained generate-parser help text).
- Cross-verb calls pinning the split design (lazy-import sites, tui_state
  precedent): `cmd_stop→cmd_finalize`, `cmd_generate→cmd_init/cmd_run/
  validate_run/cmd_finalize`, `cmd_benchmark→cmd_init`,
  `cmd_inspect→validate_run`, `cmd_run→_augment_overrides` (the last is
  resolved by housing `_augment_overrides` in the shared `cli_core` leaf).
- `git status` shows `supervisor.py`, `workers/director.py`,
  `workers/video_ltxv.py` concurrently modified — `cli.py` itself is
  clean, so the split proceeds; the supervisor-owned streaming unification
  (085) stays a proposal, not an edit.

## Fix candidates

1. Split by verb group behind existing `build_parser` seam: `cli_{parse,plan,run_ops,generate,models,inspect,status}.py` + thin `cli.py` dispatch; no behavior change, DESIGN §-ref'd modules.
2. Delete `resolve_run_dir` alias (keep one canonical); inline `_check_run_id`; import `FLOAT_DUST_EPSILON` instead of literal.
3. Repoint preset wrappers to `resolve_config` (see 083; sweep cited `cli.py:443,1228`, live tree resolves presets at `cli.py:986` + `tui_state.py:334`); unify `_augment_overrides` with `models_ensure` weight computation.
4. Gate: `gates.sh` green + `build_parser` parity tests (`test_cli_tui_split.py`) + `--help` golden diff empty.

## Progress log (2026-09-30, resolution pass)

- Re-verified live in-container (`voyage:latest`, CPU-only): `cli.py`
  2403L / 78 defs / 15 `cmd_*` (drift: +298L, −1 verb since filing).
  `git status` clean for `cli.py` (concurrent hunks only in
  supervisor/director/video_ltxv) — proceeded; captured pre-split goldens
  (`--help` × 15 verbs, `dir()` surface) to `/tmp/opencode/goldens/`.
- Wrote fail-first `tests/test_cli_split.py` (11 tests): collection failed
  pre-impl (`ModuleNotFoundError: voyage.cli_paths`, `voyage.cli_*`),
  proving the tests pin the new surface before it exists.
- Split via ephemeral AST script (`/tmp/opencode/split_modules.py`, never
  committed): verbatim block moves + free-variable import computation +
  DAG assertion. 10 new modules, all ≤365L; `cli.py` keeps the parser seam
  (`_add_*`, `build_parser`, `main`, `launch_tui`) + re-exports + `__all__`.
- Post-split regression caught by the gate suites (5 failures): cross-verb
  calls + test-patched leaves (`check_free_space`, `check_ffmpeg`,
  `_torch_available`, `_warn_if_no_cuda`, `download_*/verify_*`) resolved
  through private module refs, breaking seam patching. Fixed with the
  **seam-dispatch rule** (call-time `from voyage.cli import ...` for every
  cross-verb/patched name; top-level home imports only for never-patched
  leaves) + `__all__` explicit-export contracts on the seam (mypy
  `attr-defined` required it). All green after.
- Goldens: 15/15 `--help` byte-identical; `dir()` delta is exactly the
  removed transitive-leak names (verified: zero `from voyage.cli import
  <leaked>` / `voyage.cli.<leaked>` consumers in tree); registry + help
  re-verified after the dispatch fix. Live smoke: `init + status +
  validate` on a scratch fake run, exit 0.
- Gates on touched files: `ruff check` + `ruff format --check` clean,
  `mypy strict` clean (15 files). New tests 27/27 (incl. a mid-pass TDD
  correction: unresolved `VideoConfig` carries ltxv `segment_frames` —
  planning tests must `with_video_backend`-resolve first).
- Related suites: cli_hardening/cli_tui_split/cli_validate_handoff/
  backend_registry/config_resolution/augment_config/registry_pins/hashing/
  tui_state/cuda_preflight/director_default/generate/console/benchmark/
  observability/av_alignment/commit_hardening/generate_ensure (298) +
  tui_app (33) + 213 more across 18 suites — all green. One foreign
  failure: `test_qualify_sh_fails_closed_without_nvidia_smi` (concurrent
  `scripts/qualify.sh` + `scripts/lib/common.sh` work; empty-PATH
  `dirname` failure, zero Python involvement).

## Resolution

- Verdict: **fixed**. `cli.py` 2403 → **735L** (parsers ~460 + `__all__`
  114 + re-exports + `main`); verbs live in `cli_paths` (75) /
  `cli_planning` (194) / `cli_core` (50) / `cli_run_ops` (160) /
  `cli_models` (315) / `cli_status` (338) / `cli_validate` (253) /
  `cli_finalize` (149) / `cli_generate` (228) / `cli_observe` (365).
- Residual: parsers stay in the seam by design (they bind `cmd_*` at
  `set_defaults` time — moving them would need the seam trick in reverse).
  `cli.py` at 735L is the documented stable point, not a new god module
  (zero generation logic; verbs are one import away).
- Standing rule for future verb work (enforced by `test_cli_split.py` +
  the seam comment in `cli.py`): cross-verb + patched names resolve via
  `from voyage.cli import ...` at call time; never bind them from home
  modules, or seam patching silently stops intercepting.

## Refs

- Issues 025 (registries), 036 (split signal), 020 (TOML escaper duality for the same file)
- AGENTS.md §12 split signal; `tests/test_cli_hardening.py` (25), `tests/test_cli_tui_split.py` (10)
