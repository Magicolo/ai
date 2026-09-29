# 012 — Unpinned base images and toolchain: no digests, unversioned apt/ffmpeg/pip

- Status: resolved (verified 2026-09-25: digests + apt/ffmpeg/pip pins)
- Severity: major (reproducibility / supply chain)
- Area: containers — all three Dockerfiles
- Rank rationale: finalize bytes are not reproducible across rebuilds; base-image
  drift is silent.

## Technical description

```
Voyage/Dockerfile:1:FROM python:3.12-slim
Voyage/worker/Dockerfile.director:1:FROM python:3.12-slim
Voyage/worker/Dockerfile.video:1:FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
```

(`rg ^FROM` output verified live by orchestrator 2026-09-25 — no `@sha256`
digest on any FROM.) `apt-get install ffmpeg / python3.10-dev / git / gcc`
unversioned; `pip install --upgrade pip` floats pip itself. `ffmpeg` version drift
changes encode output (libx264 defaults, filter semantics) while `run_manifest.json`
records only `{"python": ...}` (see `cli.py:cmd_init`).

## Why this is an issue

No digest on any FROM and unversioned `ffmpeg`/`pip` mean today's verified
image and next month's identically-tagged rebuild can differ silently,
including libx264 encode behavior that changes the very bytes the pipeline
checksums. `run_manifest.json` records only the Python version, so there is no
provenance trail tying an artifact back to the toolchain that wrote it. For a
pipeline built on per-segment checksums, leaving the writer of those bytes
unpinned is a gap in the integrity story. Anyone comparing runs across
rebuilds or auditing where a final MP4 came from pays for the missing pins.

## Evidence

`rg -n "^FROM|apt-get install|upgrade pip" Voyage/Dockerfile
Voyage/worker/Dockerfile.*` → all floating (sweep output above).

## Reproduction

Rebuild at two dates; `ffmpeg -version` and base-layer digests differ while the
tag (`:latest`) is identical (see also 054).

## Source references

- `Voyage/Dockerfile:1,5`; `Voyage/worker/Dockerfile.video:1,13-16,18,33-35`;
  `Voyage/worker/Dockerfile.director:1,13-15,17`.

## Resolution candidates

1. `FROM python:3.12-slim@sha256:<…>` (and CUDA equivalent) in all three files.
2. `apt-get install ffmpeg=<ver>` (or record `ffmpeg -version` into
   `run_manifest.json`); pin pip (`pip==25.x`).
3. Document the digest-bump procedure (where digests live, how to update).

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; FROM lines re-verified live.
- Open: pin digests + versions.
- 2026-09-25 (repair pass): added `## Why this is an issue`; FROM/apt/pip
  lines re-verified live, still floating; no other change.
- 2026-09-29 (this track, scope Dockerfiles): re-read live — ADOPTED, no
  duplicate work. FIXED by another track: all three FROM carry digests
  (`Dockerfile:8`, `worker/Dockerfile.video:7`,
  `worker/Dockerfile.director:4`); `ffmpeg=7:7.1.5-0+deb13u1` pinned
  slim + director; `pip==25.0.1` pinned all three; bump procedures in
  header comments + `docs/INSTALL.md`. Remainder documented in-tree:
  video apt set stays unversioned (`Dockerfile.video:23-27`, digest-pinned
  FROM is the guarantee; freeze on next GPU build). No edit in this track.
  Note: full video/director image builds skipped per task (slim build via
  `gates.sh` only).
