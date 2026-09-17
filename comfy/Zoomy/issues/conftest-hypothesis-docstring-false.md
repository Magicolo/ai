# `conftest.py` docstring promise is contradicted by `.hypothesis/` on disk

- Severity: medium (process hygiene — the tree already shows the failure).
- Status: verified open. `Zoomy/tests/conftest.py:1-13`,
  `Zoomy/.hypothesis/` on disk (root-owned `constants/`, `unicode_data/`).

## Evidence

The docstring says disabling the example database means "test runs never
write `.hypothesis/` residue anywhere (in particular not into the
bind-mounted source tree)". False on both halves: only the *example DB*
is disabled — Hypothesis still writes `constants` + `unicode_data`
caches — and `Zoomy/.hypothesis/` exists in the tree right now,
root-owned from a container-as-root bind-mount run (host cleanup needs
`sudo`). Separately, `database=None` trades away CI failure replay
(unreproducible shrinking across runs) for that hygiene.

## Fix (pick one, document it)

- Option A (hermetic): keep `database=None` AND set
  `HYPOTHESIS_STORAGE_DIRECTORY=/tmp/…` (or equivalent env in both
  scripts and `Dockerfile`), fix the docstring to describe *both*
  mechanisms, and delete the on-disk `.hypothesis/` from inside the
  container.
- Option B (replayable): restore the database into `/tmp` (keeps replay
  within a run, nothing in the tree) and document the trade-off.

Either way the docstring must describe what is actually configured, and
`.hypothesis/` must be in `.dockerignore`/`.gitignore` as a backstop
(see `gitignore-dockerignore-hygiene.md`).

## Verification

- In-container gate run leaves `git status --porcelain` clean of
  `.hypothesis/`; docstring matches the mechanism.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
