# 230 — Image reproducibility drift: `ffmpeg 7.1.5` (slim) vs `4.4.2` (video) + unpinned `apt` and the deadsnakes PPA in `Dockerfile.ltx`

Severity: MEDIUM (track E-10).

## Technical description

`Dockerfile:15` pins `ffmpeg=7:7.1.5-0+deb13u1` (trixie);
`worker/Dockerfile.video:24` pins `ffmpeg=7:4.4.2-0ubuntu0.22.04.1` (jammy) with a comment
acknowledging the gap; `worker/Dockerfile.ltx:33-42` installs `python3.11 … ffmpeg
curl …` fully unpinned with "intentionally unpinned … digest keeps it reproducible". The
slim header (`Dockerfile:1-7`, issue 012) claims pins freeze "ffmpeg/libx264 encode
behavior".

## Rationale

Finalize mux/encode runs in slim (`media.py`) and model-pass decode in video/ltx images.
Different ffmpeg/libx264 majors can shift bytes for identical inputs — the manifest
records no ffmpeg version, so two runs with identical configs can ship different
`final.mp4` bytes with no provenance trail.

## Live evidence

```
$ grep -n "ffmpeg=" Dockerfile worker/Dockerfile.video
Dockerfile:15: ffmpeg=7:7.1.5-0+deb13u1
worker/Dockerfile.video:24: ffmpeg=7:4.4.2-0ubuntu0.22.04.1
$ sed -n '33,42p' worker/Dockerfile.ltx   # no = pins
```

Repro: `docker run --rm voyage:latest ffmpeg -version | head -1` vs
`docker run --rm --entrypoint ffmpeg voyage-video:latest -version | head -1` →
different majors.

## Source refs

`Dockerfile:1-16`; `worker/Dockerfile.video:1-32`; `worker/Dockerfile.ltx:1-42`;
`docs/INSTALL.md:7-30`.

## Online sources

- Docker build best practices (pin + digest, reproducible installs).

## Fix candidates

- Pin ltx apt (`dpkg-query -W` re-query procedure already documented in the video
  header); either align ffmpeg majors or record per-stage `ffmpeg_version` (doctor already
  captures `_ffmpeg_version`) into `manifest.presentation`/`finalize_completed` metrics.

## Log

- 2026-10-07: filed from read-only Track E sweep; no code touched.

## Consolidated from 233_ltx_dockerfile_unpinned_apt_deadsnakes_ppa (2026-10-07)

### Technical description (from 233)
`Dockerfile.video` pins every apt row (`python3.10=3.10.12-…`, `ffmpeg=7:4.4.2-…`,
`Dockerfile.video:19-25,63-65`). `Dockerfile.ltx` installs `software-properties-common`,
`add-apt-repository ppa:deadsnakes/ppa`, then `python3.11 python3.11-venv python3.11-dev
git gcc g++ ffmpeg curl ca-certificates` with **no `=` versions**, explicitly deferring
("noble versions were not pre-resolved", `Dockerfile.ltx:24-27`).

### Rationale (from 233)
apt-source compromise or snapshot drift → arbitrary root code at build.

### Live evidence (from 233)
`worker/Dockerfile.ltx:33-42` vs `worker/Dockerfile.video:19-25`.
`grep -n "apt-get install" worker/Dockerfile.ltx` shows zero `=` pins.

Repro: rebuild `voyage-ltx` twice across a noble snapshot roll → `dpkg-query -W` deltas
with no Dockerfile diff (contrast the video image's re-query procedure,
`Dockerfile.video:26-32`).

### Source refs (from 233)
`worker/Dockerfile.ltx:24-42`.

### Online sources (from 233)
- Docker best practices (version-pinning `package=1.3.*` as cache-busting +
  reproducibility; `--no-install-recommends` already correct).
- Ubuntu PPA trust model (PPA adds a new signing key + full archive for the build).

### Fix candidates (from 233)
- Run the documented re-query (`dpkg-query -W`) once and pin every row `=` like the
  video image; pin the PPA key fingerprint or vendor a python3.11 base instead of adding
  an archive mid-build.

### Log (from 233)
- 2026-10-07: filed from read-only Track F sweep; no code touched.
