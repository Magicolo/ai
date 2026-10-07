# 244 — Dead `_preset_int` helper; `models_dir` hardcoded `/models` on every backend switch; no CLI surface for model roots

Severity: MEDIUM (track B-08).

## Technical description

`_preset_int` (config.py:720) has zero callers. `resolve_config` backend branch rebuilds
audio/sfx with `"models_dir": "/models"` hardcoded, clobbering any customized root. All
three `models_dir` fields default `/models` with no `--models-dir` flag, so the hardcode
is a no-op today and a trap tomorrow.

## Rationale

Dead code + silent clobber on a path every backend switch executes. The XDG question
(where is `$HOME/.cache` vs `$XDG_CACHE_HOME`?) is decided in `run.sh:139`
(`${VOYAGE_MODELS:-$HOME/.cache/voyage-models}`) with no config counterpart.

## Live evidence

```
_preset_int refs: voyage/config.py:720 only (def, zero calls)
"models_dir": "/models" in resolve_config: voyage/config.py:807,809
models_dir fields: config.py:342,433,527 (all default "/models", no CLI flag references)
models="${VOYAGE_MODELS:-$HOME/.cache/voyage-models}"  # scripts/run.sh:139
```

Repro: set `video.models_dir=/custom` programmatically → `resolve_config(cfg,
backend="ltxv")` → `/models` (AST shows the literal overwrite; no test pins
preservation).

## Source refs

`voyage/config.py:720-726,799-809`.

## Online sources

- `https://specifications.freedesktop.org/basedir/latest/` (`$XDG_CACHE_HOME` default
  `$HOME/.cache`).

## Fix candidates

- Delete `_preset_int` or use it; preserve `models_dir` across backend switches (only
  default when unset); add `--models-dir` or document why roots are image-fixed; honor
  `${XDG_CACHE_HOME:-$HOME/.cache}/voyage-models` in `run.sh`.

## Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.

## Evaluation (2026-10-07, resolution pass)
- Still relevant, half only: `_preset_int` (`voyage/config.py:720`) still has
  zero callers live (repo-wide grep: the `def` is the sole hit outside this
  issue file + index — no test, worker, or CLI references it), so deletion is
  safe. The `models_dir` hardcode (`resolve_config`, `"/models"` on the audio
  + sfx rebuilds) is confirmed live but admitted no-op (all three defaults are
  already `"/models"`, no `--models-dir` flag exists) — removal would change
  no behavior today but risks masking the trap tomorrow, so per the task brief
  it stays as code with a warning comment instead.
- Note: the finding header says MEDIUM while the task brief treats this as
  LOW — the dead-helper half is LOW (no runtime effect); only the silent-
  clobber half carries MEDIUM weight, and only once a models-dir override
  path exists.

## Resolution (2026-10-07)
- Code changes (`voyage/config.py` only): deleted the dead `_preset_int`
  helper (5 lines, zero callers — verified by repo-wide grep before removal);
  left both `"models_dir": "/models"` hardcodes untouched and added a
  5-line comment at the backend-switch branch documenting the no-op-today /
  clobber-tomorrow trap with the preserve-custom-roots precondition.
- Status: RESOLVED (dead code removed; hardcode deliberately kept + now
  documented). Verification: scoped pytest + ruff in-container
  (see verification below). No caller updates needed (there were none);
  `resolve_config` behavior byte-identical (comment-only delta on that path).

## Verification (2026-10-07, post-fix)
- `grep -rn _preset_int Voyage/voyage Voyage/tests`: def gone, zero refs.
- `./scripts/test.sh -m "not gpu" tests/test_registry_pins.py
  tests/test_interp_backend.py tests/test_backend_registry.py` →
  59 passed, 1 skipped (skip is pre-existing: torch-bearing leg, slim image).
- In-container `ruff check` + `ruff format --check` on touched files: clean.
  In-container `mypy voyage`: no issues in 97 files.
- No GPU workloads run.

## Consolidated from 228_dead_presetint_modelsdir (2026-10-07)

No unique content — 228 was a duplicate copy of this exact finding: identical technical description, rationale, live evidence, repro, source refs, online sources, and fix candidates; only the title's severity spelling and log wording differed. Nothing to fold.
