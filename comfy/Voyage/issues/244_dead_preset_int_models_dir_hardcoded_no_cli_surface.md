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

## Consolidated from 228_dead_presetint_modelsdir (2026-10-07)

No unique content — 228 was a duplicate copy of this exact finding: identical technical description, rationale, live evidence, repro, source refs, online sources, and fix candidates; only the title's severity spelling and log wording differed. Nothing to fold.
