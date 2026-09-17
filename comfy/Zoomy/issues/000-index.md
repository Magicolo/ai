# Zoomy issue index

Remaining open issues. Resolved issues were fixed, committed, and removed
(2026-09-17): both HIGHs (pad-empty-hang, single-frame-finalize-index-error)
and 14 MEDIUMs (popen-stdin-write-leak, silent-video-cleanup-no-finally,
pil-handles-unclosed, silent-model-fallbacks, settings-whitespace-text,
find-family-uncaught-handlers, safe-statistics-narrow-except,
clear-frames-traversal-ignore-errors, unguarded-public-math,
dockerfile-reproducibility, requirements-gpu-mixed-pins,
pyproject-missing-enforcement) plus gitignore-dockerignore-hygiene and
conftest-hypothesis-docstring-false (verified complete). Full record in git
history. Fixes land hardest-first; each fix is TDD'd, gated
(`Zoomy/scripts/quality-gates.sh` green), and committed per issue.

## Open issues

### Medium

- `scripts-hardening.md` — remaining: advisory shellcheck pass (no
  shellcheck available in this environment to verify with).

### Low

- `magic-sync-constants.md` — literal `25`/`44100` duplicate
  `OUTPUT_SAMPLE_RATE`.
- `system-memory-zero-not-na.md` — `/proc` failure reports `0.0 GiB`, not
  `n/a`.
- `vendor-compat-no-lock.md` — `_APPLIED_PATCHES` mutated without a lock.
- `loop-stop-global-event.md` — module-level stop flag shared across sessions.
- `interrupt-clear-race.md` — `self._interrupt.clear()` can wipe a
  concurrently set interrupt.
- `coercion-silent-defaults.md` — garbage payloads coerce to `0.0`/`set()`.
- `output-residue-cleanup.md` — e2e mp4s accumulating under `Zoomy/output/`.
