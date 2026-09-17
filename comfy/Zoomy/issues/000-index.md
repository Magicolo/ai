# Zoomy issue index

All 24 filed issues are resolved (2026-09-17): fixed, gated
(`Zoomy/scripts/quality-gates.sh` green), committed per issue, and removed.
Full record in git history.

- HIGH (2): pad-empty-hang, single-frame-finalize-index-error.
- MEDIUM (14): popen-stdin-write-leak, silent-video-cleanup-no-finally,
  pil-handles-unclosed, silent-model-fallbacks, settings-whitespace-text,
  find-family-uncaught-handlers, safe-statistics-narrow-except,
  clear-frames-traversal-ignore-errors, unguarded-public-math,
  dockerfile-reproducibility, requirements-gpu-mixed-pins,
  pyproject-missing-enforcement, conftest-hypothesis-docstring-false,
  scripts-hardening.
- LOW (8): gitignore-dockerignore-hygiene, magic-sync-constants,
  system-memory-zero-not-na, vendor-compat-no-lock, loop-stop-global-event,
  interrupt-clear-race, coercion-silent-defaults, output-residue-cleanup.

Fixes land hardest-first; each fix is TDD'd (failing test first, watched
red), gated green, reviewed, and committed per issue.
