# Helper scripts need hardening (quoting, cache env, Hypothesis dir)

- Severity: medium (the scripts ARE the quality loop — gaps here leak
  root-owned residue into the host tree).
- Status: partially fixed 2026-09-16 (arg forwarding hardened to
  `sh -c '…' sh "$@"`, `HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis`
  exported by both scripts). Remaining: shellcheck advisory pass;
  `run-tests.sh` needs no ruff/mypy env (it runs pytest only).

## Evidence

1. **Unquoted `$*`** in both scripts (`… pytest -p no:cacheprovider $*`):
   word-splitting on test-path args (SC2086). `"$@"` preserves arguments.
2. **`run-tests.sh` sets no cache env**: `quality-gates.sh` exports
   `RUFF_CACHE_DIR=/tmp/ruff-cache` + `MYPY_CACHE_DIR=/tmp/mypy-cache`,
   but `run-tests.sh` runs bare `pytest` — plus neither script sets
   `HYPOTHESIS_STORAGE_DIRECTORY`, so every test run can write
   root-owned `.hypothesis/` through the bind mount (it already has —
   see `conftest-hypothesis-docstring-false.md`).
3. **No `set -o pipefail` gap**: present (`set -euo pipefail`) — good,
   keep.
4. Scripts are exempt from testing per the working agreement, but they
   must still pass `ruff`/`mypy`-adjacent scrutiny where applicable
   (shellcheck-clean) and uphold the same standards.

## Fix

- `"$@"` in both scripts; add the two `RUFF/MYPY_CACHE_DIR` exports (they
  cost nothing) and `HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis-…` to
  both; document that caches never land in the tree.
- Run `shellcheck` on both scripts once (advisory, not gated — no new
  toolchain on the host).

## Verification

- `./Zoomy/scripts/run-tests.sh -q tests/test_settings.py "arg with space"`
  passes args intact; in-container run leaves no root-owned cache dirs in
  the tree (`git status --porcelain` clean).
- Gates: `Zoomy/scripts/quality-gates.sh` green.
