# 084 — No `.dockerignore`: every build ships ~400 MB `output/` as context

- Status: open
- Severity: low-medium (slow builds; cache churn)
- Area: containers — `Voyage/` (absent file) + `Voyage/Dockerfile:9-11`
- Rank rationale: pass-2 finding; quantifies 054's hygiene note (395–406 MB
  measured).

## Technical description

`ls Voyage/.dockerignore` → nonexistent (verified live by orchestrator
2026-09-25). `du -sh Voyage/output/` → `395M` (sweep) / `406M` (orchestrator —
grows with renders). Docker sends the full context before applying `COPY`, so
each `scripts/*.sh` build uploads the 400 MB frames/videos plus
`.mypy_cache`/`.ruff_cache`/`issues/` even though the image only needs
`pyproject.toml`/`README.md`/`voyage/`/`tests/`.

## Why this is an issue

Every docker build uploads hundreds of megabytes of render artifacts as build
context before any `COPY` runs — slow builds and cache churn on every render,
for files the image never uses. A one-file `.dockerignore` removes the whole
class, and the output dir only grows (395M → 406M → 418M across verifications).

## Evidence

```
$ ls Voyage/.dockerignore
ls: cannot access 'Voyage/.dockerignore': No such file or directory
$ du -sh Voyage/output/
406M	Voyage/output/
```

Re-checked 2026-09-25: still absent; `du -sh Voyage/output/` → `418M`
(grows with renders).

## Reproduction

Commands above; `docker build` with `DOCKER_BUILDKIT=1` shows transferred context
size.

## Source references

- `Voyage/Dockerfile:9-11`; `Voyage/worker/Dockerfile.*` COPY lines.

## Resolution candidates

Add `.dockerignore` excluding `output/`, `.git`, `*_cache/`, `issues/`,
`reports/` (reports are docs, not build inputs). Pairs with 054's COPY reorder.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep; absence + size re-verified live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  absence live (still no `.dockerignore`); `output/` now `418M` (was 406M).
- Open: add `.dockerignore`.
