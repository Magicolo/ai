# 087 — Lint/type targets (py312) exceed the video runtime (py3.10): latent 3.11+ rewrite hazard

- Status: open
- Severity: low (next "cleanup" can break the video image with no gate catching it)
- Area: toolchain — `Voyage/pyproject.toml:33-35,40-44`,
  `Voyage/worker/Dockerfile.video:15` (`python3.10`)
- Rank rationale: pass-2 finding; the skew already bit once (two live
  workarounds in-tree).

## Technical description

`target-version = "py312"` + `mypy python_version = "3.12"` while the video
worker runs 3.10. Proof the skew already bites: `voyage/concepts.py:75` and
`voyage/persistence.py:28` both carry `# noqa: UP017` workarounds because ruff
wants `datetime.UTC` (3.11+) on a 3.10 runtime. The next contributor who "fixes"
a flagged idiom to 3.12 syntax breaks the video image with no gate catching it.

## Why this is an issue

Lint and type targets promise py312 idioms while the video worker runs 3.10,
so the next well-meaning "modernize this flagged line" cleanup breaks the
largest image with no gate catching it. The skew already bit (the in-tree
`UP017` workarounds exist because ruff wants 3.11+ syntax on a 3.10 runtime) —
every new workaround is a symptom, and the hazard grows with each one.

## Evidence

Cited lines verified live (`pyproject.toml:33-35` ruff `py312`,
`:40-44` mypy `3.12`, `Dockerfile.video:15` py3.10). `UP017` workarounds
(re-run 2026-09-25 — now THREE sites, not two):

```
$ rg -n "UP017" Voyage/voyage/*.py
Voyage/voyage/logrotate.py:11:Note: the `# noqa: UP017` marks below are deliberate — ...
Voyage/voyage/logrotate.py:28:    return datetime.datetime.now(datetime.timezone.utc).date()  # noqa: UP017
Voyage/voyage/logrotate.py:47:            tz=datetime.timezone.utc,  # noqa: UP017
Voyage/voyage/concepts.py:75:    ...  # noqa: UP017 — worker image is py3.10, datetime.UTC needs 3.11+
Voyage/voyage/persistence.py:28:        ...  # noqa: UP017 — worker image is py3.10, datetime.UTC needs 3.11+
```

## Reproduction

`ruff check --target-version py312 voyage/` vs `--target-version py310`
flag-diff on the two lines.

## Source references

- Files/lines above.

## Resolution candidates

Set `target-version = "py310"` and `mypy python_version = "3.10"` (the minimum in
`requires-python`), or document why 3.12 was chosen.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (targets + video py3.10 match); note: `UP017` workarounds
  are now THREE sites (`logrotate.py:28,47` plus the two cited) — the hazard
  described here has already grown once more.
- Open: retarget or document.
