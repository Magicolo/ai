# 076 — Unversioned `apt` set in the video image while slim/director pin `ffmpeg`

- Severity: MEDIUM
- File: `Voyage/worker/Dockerfile.video:20-22` (`python3.10 python3.10-venv python3-pip git gcc g++ ffmpeg` + later `python3.10-dev` at `:60-61`, all unversioned); contrast `Voyage/Dockerfile:15` (`ffmpeg=7:7.1.5-0+deb13u1`) — the former `Voyage/worker/Dockerfile.director:17` pin cited by the sweep no longer exists (director image file deleted upstream; needs orchestrator confirmation)
- Area: containers — video image apt reproducibility

## Description

The CUDA image's entire apt layer floats; the file admits it (`:23-27`: "stays unversioned on purpose… Next GPU build should freeze"). The digest-pinned `FROM` softens this (same snapshot → same apt versions *today*), but any `FROM` bump or repo-side rebuild silently changes compiler/CUDA-adjacent toolchain + ffmpeg encode bytes, and `run_manifest.json` cannot record it (the slim header `:1-7` documents exactly this risk class).

## Rationale

Reproducible-media + supply-chain hygiene: unpinned apt is the largest unaudited blob in the GPU image, and `ffmpeg` version directly changes output bytes the pipeline checksums.

## Live evidence

Re-verified 2026-09-30 live:

```
Voyage/worker/Dockerfile.video:20-22:
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git gcc g++ ffmpeg \
    && rm -rf /var/lib/apt/lists/*
Voyage/worker/Dockerfile.video:60-61:
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10-dev \
    && rm -rf /var/lib/apt/lists/*
Voyage/Dockerfile:15:
    && apt-get install -y --no-install-recommends ffmpeg=7:7.1.5-0+deb13u1 \
```

`ls Voyage/worker/` shows only `Dockerfile.video` — the director pin cited above is gone with the file. Slim/director lines show the established pin pattern not applied in the video image. `apt-get install` appears twice in the video file with no `=` pins.

## Repro

`docker run --rm voyage-video:latest ffmpeg -version; dpkg -l | grep -E "ffmpeg|gcc|python3.10"` across two builds separated by a `FROM` bump → versions move with no tree diff.

## Fix candidates

1. On the next GPU build, `apt-cache policy …` → exact `pkg=ver` pins for all seven packages (procedure already documented in the slim header `Dockerfile:5-7`).
2. Record versions in `docs/INSTALL.md` per the video header's own bump procedure (`worker/Dockerfile.video:5-6`).

## Refs

- Reproducible-builds apt pattern (`curl=8.7.1-r0 …`) — https://www.systemshardening.com/articles/cicd/reproducible-builds/

## Resolution log 2026-09-30 (Rank-2 batch)

Re-verified live 2026-09-30 — CONFIRMED with one stale citation: the
director pin (`Dockerfile.director:17`) no longer exists;
`ls Voyage/worker/` shows only `Dockerfile.video` (director unified into
the video image; the `voyage-director:latest` image is still local but its
Dockerfile is retired). Slim pin pattern confirmed (`Dockerfile:15`
`ffmpeg=7:7.1.5-0+deb13u1` + header procedure `:1-7`/`:11-13`). The video
file had 2 unpinned apt RUNs covering 8 packages.

Fix (`Voyage/worker/Dockerfile.video` only): exact versions recovered
CPU-only — `docker run --rm voyage-video:latest dpkg-query -W ...`
against image 63b8c37d (the same image the pip freeze cites), so no GPU
build was needed. All 8 pinned: python3.10(-venv,-dev)
`3.10.12-1~22.04.18`, python3-pip `22.0.2+dfsg-1ubuntu0.7`, git
`1:2.34.1-1ubuntu1.17`, gcc/g++ `4:11.2.0-1ubuntu1`, ffmpeg
`7:4.4.2-0ubuntu0.22.04.1`. Jammy snapshot, hence older than the slim
trixie pins (ffmpeg 4.4.2 vs 7.1.5) — expected, different distro
snapshots. Freeze provenance + bump procedure recorded in the Dockerfile
note. No dead legs.

Tests: pin test in `Voyage/tests/test_containers_rank2.py` (all 8
`pkg=ver` substrings at the frozen versions + the unversioned NOTE gone;
deliberately brittle — bumps update both files). Batch 48 passed; full
suite 1388 passed, 5 skipped, 1 deselected.

Residuals: (1) versions recorded in Dockerfile + test only, NOT in
`docs/INSTALL.md` (outside owned scope — the Dockerfile note points at
the INSTALL step; owning pass should copy the table); (2) pins not
rebuild-proven (digest-pinned FROM makes resolution near-certain, but
unverified until the next video build).

DESIGN proposals: none.
