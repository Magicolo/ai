# 054 — Image layer hygiene: `COPY voyage/` invalidates pip layer; `tests/` baked into prod images; `:latest` tags; no `.dockerignore`

- Status: open
- Severity: medium (slow iteration; test code ships in prod; stale-vs-bind
  confusion)
- Area: containers — all Dockerfiles, `run.sh`
- Rank rationale: every one-line edit reinstalls the package layer; two behaviors
  from one tag.

## Technical description

- `Voyage/Dockerfile:9-11`, `worker/Dockerfile.video:87-88`,
  `worker/Dockerfile.director:29-31`: `COPY voyage/` precedes
  `pip install -e .`, so every one-line code edit reinstalls the package layer.
- `tests/` is `COPY`d into the slim + director images (video image correctly
  omits it) — test code ships in production, widening what auditors must review;
  no `.dockerignore` exists to exclude `.mypy_cache/.ruff_cache/output/reports`.
- `run.sh` bind-mounts `$PWD:/app`, so dev runs shadow baked code (stale-image
  confusion), while detached runs use stale baked code — two behaviors from one
  mutable `:latest` tag. `git ls-files Voyage/reports Voyage/output` confirms
  reports tracked, `output/` ignored (correct — keep).

## Why this is an issue

Slow iteration compounds: every one-line `voyage/` edit reinstalls the
package layer, turning a 10-second rebuild into minutes across three images
— and developers quickly learn to skip rebuilds, which is exactly how
stale-image confusion ("it works in dev, fails detached") starts. Shipping
`tests/` in production images widens the audit surface for no runtime
benefit, and the mutable `:latest` tag means two runs of the "same" image
can execute different code. Blast radius is developer velocity + release
reproducibility; the fix is Dockerfile ordering plus a `.dockerignore`.

## Evidence

```
$ rg -n "^COPY" Voyage/Dockerfile Voyage/worker/Dockerfile.director Voyage/worker/Dockerfile.video
Dockerfile:9:COPY pyproject.toml README.md ./
Dockerfile:10:COPY voyage/ ./voyage/
Dockerfile:11:COPY tests/ ./tests/
worker/Dockerfile.video:87:COPY pyproject.toml README.md /app/
worker/Dockerfile.video:88:COPY voyage/ /app/voyage/
worker/Dockerfile.director:29:COPY pyproject.toml README.md /app/
worker/Dockerfile.director:30:COPY voyage/ /app/voyage/
worker/Dockerfile.director:31:COPY tests/ /app/tests/
$ git check-ignore -v output
comfy/Voyage/.gitignore:6:output/	output
```

## Reproduction

Time a rebuild after a one-line `voyage/` edit (full reinstall); `docker run`
without the bind-mount and observe stale code.

## Source references

- Files/lines above.

## Resolution candidates

Install deps first (`COPY pyproject.toml` → `pip install` → `COPY voyage/`);
drop `COPY tests/` from runtime images (gates already mount the tree); add
`.dockerignore`; tag images by content hash instead of `:latest`.

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep.
- 2026-09-25 (repair): re-verified COPY lines live (Evidence pasted) —
  corrected ranges (`Dockerfile:9-11`, video `:87-88`, director `:29-31`;
  the old `:9-13/:87-90/:29-33` overshot). Confirmed `output/` ignored via
  `git check-ignore`. Added `## Why this is an issue`.
- Open: reorder COPY + dockerignore + tagging.
