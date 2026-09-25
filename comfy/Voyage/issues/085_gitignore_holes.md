# 085 — `Voyage/.gitignore` holes: root-level media/log artifacts are committable

- Status: open
- Severity: low (stray artifacts committable at the Voyage root)
- Area: repo hygiene — `Voyage/.gitignore:1-6`
- Rank rationale: pass-2 finding; 042 covered Zoomy-side caches, not Voyage
  gitignore coverage.

## Technical description

`.gitignore` holds only `__pycache__/ *.pyc .mypy_cache/ .ruff_cache/
.pytest_cache/ output/`. `git check-ignore -v final.mp4 run/audio.wav model.pt
metrics.jsonl coverage.xml .coverage` from `Voyage/` → only `.coverage` matches
(via the *parent* `comfy/.gitignore`, not Voyage's). A misdirected
`finalize --output final.mp4`, stray `metrics.jsonl`, `*.pt`, `*.log`, or
`coverage.xml` at the Voyage root would be committed silently.

## Why this is an issue

A single misdirected `--output` or stray artifact at the Voyage root is
silently committable — including multi-hundred-megabyte media files that then
live in git history forever. The Voyage `.gitignore` covers caches and
`output/` but none of the artifact extensions the tool itself produces, so the
repo's most likely accidents are exactly the ones not ignored.

## Evidence

`check-ignore` output (re-run 2026-09-25 from `comfy/`):

```
$ git check-ignore -v Voyage/final.mp4 Voyage/run_audio.wav Voyage/model.pt Voyage/metrics.jsonl Voyage/coverage.xml Voyage/.coverage
comfy/.gitignore:11:.coverage	Voyage/.coverage
```

Only `.coverage` matches (via the parent `comfy/.gitignore`, not Voyage's);
every media/log artifact is committable. `Voyage/.gitignore:1-6` holds only
caches + `output/` (verified live).

## Reproduction

From `comfy/`: `git check-ignore -v Voyage/final.mp4` → no match (exits 1);
or `touch Voyage/final.mp4 && git status --porcelain -- Voyage/final.mp4` →
listed as committable.

## Source references

- `Voyage/.gitignore:1-6`.

## Resolution candidates

Append `*.mp4 *.wav *.flac *.pt *.log .coverage coverage.xml` (scoped so
`reports/*.md` and `output/` behavior are unchanged).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  live (`.gitignore:1-6` unchanged; `check-ignore` pasted above — only
  `.coverage` matches, via parent gitignore).
- Open: extend `.gitignore`.
- 2026-09-25 (fix): relevance re-verified live — `Voyage/.gitignore:1-6`
  still caches + `output/` only, so the issue was live. Appended exactly
  the candidate list in `Voyage/.gitignore:7-16` (`*.mp4 *.wav *.flac *.pt
  *.log .coverage coverage.xml`) with a scoping comment. Verified:
  `git check-ignore -v` matches all seven artifact probes via the new
  Voyage rules; `Voyage/reports/*.md` still not ignored; `output/`
  artifacts still covered by the existing `output/` rule; `git ls-files`
  shows no tracked file matching the new patterns (nothing already
  committed is affected). Note: `metrics.jsonl` (named in the technical
  description but not in the candidate list) remains committable — left
  for a follow-up decision, not silently added.

## Resolution

- Status: fixed.
- Files: `Voyage/.gitignore:7-16`.
