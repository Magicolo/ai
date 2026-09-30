# 069 — `.dockerignore` has context-bloat gaps and a stale scope comment

**Severity:** MEDIUM

**File:line:** `Voyage/.dockerignore:1-19`; evidence in tree root `ls -la`

**Area:** containers — build-context hygiene

## Description

(a) Missing entries for files demonstrably present in the tree: `.coverage` (53,248 bytes observed 2026-09-30), `coverage.xml`, `.pytest_cache/`, `.hypothesis/` (4 entries observed), `.venv/`, `*.egg-info/`, `.env`. (b) Dot-cache coverage relies entirely on the `*_cache/` glob (`:7`) — correct under Go `filepath.Match` semantics but brittle and undocumented; an explicit `.mypy_cache/ .ruff_cache/ .hypothesis/` trio is cheaper than trusting glob-dotfile behavior across builders. (c) The header comment (`:1-4`) claims "images only COPY pyproject.toml, README.md, voyage/ and tests/" — false: the slim `Dockerfile:27-31` and both worker Dockerfiles COPY `voyage/` but NOT `tests/` (deliberately, per issue 054). A stale ignore-scope comment invites someone to "fix" the ignore list to match the comment and re-bloat the context.

## Rationale

Unignored context inflates the transferred build context (the comment itself cites "~400 MB of frames/videos" for `output/`), churns layer cache on every render/probe, and can leak run artifacts (`.coverage` embeds absolute paths) into build layers.

## Evidence

Re-verified 2026-09-30 live:

```
Voyage/.dockerignore:1-19 (full file):
# Build-context hygiene (issue 084): images only COPY pyproject.toml,
# README.md, voyage/ and tests/ — everything below just inflates ...
output/
.git
*_cache/
__pycache__/
*.pyc
*.mp4
*.wav
*.flac
*.pt
*.log
issues/
reports/

Voyage/Dockerfile:27-31:
COPY pyproject.toml README.md requirements.lock ./
... (pip install) ...
COPY voyage/ ./voyage/

$ ls -la Voyage/:
.coverage (53248 bytes), .hypothesis/, .mypy_cache/, .ruff_cache/ all present,
none literally named in .dockerignore
```

`git status` noise (`.coverage` untracked-but-present) confirms gates write them into the tree.

## Repro

`docker build --progress=plain . 2>&1 | grep -i "transferring context"` with and without a probe `.coverage`/`output/*.mp4` at root; or `du -sh .coverage .hypothesis` vs context size.

## Fix candidates

1. Add `.coverage coverage.xml .pytest_cache/ .hypothesis/ .mypy_cache/ .ruff_cache/ .venv/ *.egg-info/ .env`.
2. Correct the `:1-4` comment to "voyage/ only (tests/ bind-mounted, issue 054)".
3. Consider `!`-exception audit in CI (`docker build --dry-run` context list where supported).

## Resolution log 2026-09-30 (Rank-2 batch)

Re-verified live 2026-09-30 — all three legs CONFIRMED as-read:
(a) `.coverage` (53,248 bytes), `.hypothesis/`, `.mypy_cache/`,
`.ruff_cache/` all present in `Voyage/`; `.pytest_cache/`, `.venv/`,
`*.egg-info/`, `.env`, `coverage.xml` absent today but producible by
gates/dev runs. Cross-check: `.gitignore` (issue 085) already lists
`.coverage`, `coverage.xml`, `.pytest_cache/` — `.dockerignore` lagged it.
(b) `*_cache/` does match dot-caches under Go `filepath.Match` (no
shell-style dotfile exception), but that reliance was undocumented.
(c) Header false on two counts: NEITHER image COPYs `tests/` (slim
`Dockerfile:27-31` COPYs pyproject/README/requirements.lock + `voyage/`;
video `Dockerfile.video:189-190` COPYs pyproject/README + `voyage/`), and
the comment omits `requirements.lock`.

Fix (`Voyage/.dockerignore` only): 9 patterns added
(`.mypy_cache/ .ruff_cache/ .hypothesis/ .pytest_cache/ .coverage
coverage.xml .venv/ *.egg-info/ .env`), header rewritten to the true COPY
graph with the issue-054 bind-mount note. No dead legs.

Tests: 3 scan tests in `Voyage/tests/test_containers_rank2.py`
(pytest/coverage entries, dev-cache/env entries, scope-comment). Batch
48 passed; full in-container suite 1388 passed, 5 skipped, 1 deselected.

Residuals: none material. DESIGN proposals: none (patterns + comment only).
