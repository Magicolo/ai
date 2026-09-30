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

## Fix candidates

1. Split by verb group behind existing `build_parser` seam: `cli_{parse,plan,run_ops,generate,models,inspect,status}.py` + thin `cli.py` dispatch; no behavior change, DESIGN §-ref'd modules.
2. Delete `resolve_run_dir` alias (keep one canonical); inline `_check_run_id`; import `FLOAT_DUST_EPSILON` instead of literal.
3. Repoint preset wrappers to `resolve_config` (see 083; sweep cited `cli.py:443,1228`, live tree resolves presets at `cli.py:986` + `tui_state.py:334`); unify `_augment_overrides` with `models_ensure` weight computation.
4. Gate: `gates.sh` green + `build_parser` parity tests (`test_cli_tui_split.py`) + `--help` golden diff empty.

## Refs

- Issues 025 (registries), 036 (split signal), 020 (TOML escaper duality for the same file)
- AGENTS.md §12 split signal; `tests/test_cli_hardening.py` (25), `tests/test_cli_tui_split.py` (10)
