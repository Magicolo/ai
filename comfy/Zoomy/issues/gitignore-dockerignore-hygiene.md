# `.gitignore` / `.dockerignore` hygiene

- Severity: low (fragile defaults — harmless today, footgun on any `COPY .`).
- Status: verified open. `.gitignore` (4 lines), `Zoomy/.dockerignore`
  (6 lines).

## Evidence

Root `.gitignore` ignores only tokens + `Zoomy/output/*` — missing
`__pycache__/`, `*.pyc`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
`.venv/`, `.coverage`, `.DS_Store`. The `.hypothesis` case is covered
only by Hypothesis's *auto-generated* `Zoomy/.hypothesis/.gitignore`
(`*`, repo-unowned) — works per `git check-ignore` but is not a conscious
repo rule. `Zoomy/.dockerignore` misses `.hypothesis/`, `.git/`,
`output/`, `*.mp4/*.flac/*.wav`, `.venv/`, `.coverage` — harmless while
the Dockerfile uses explicit `COPY` lines, fragile the moment anyone
writes `COPY .`.

## Fix

Append the standard cache/venv/coverage entries to root `.gitignore`
(including an explicit `Zoomy/.hypothesis/`), and the residue/media
entries to `Zoomy/.dockerignore`. Verify `git check-ignore` on each and
that `docker build` context shrinks or stays equal.

## Verification

- `git status --porcelain` stays clean after a bare `pytest`/`mypy`/`ruff`
  run; `git check-ignore -v` hits the new rules.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
