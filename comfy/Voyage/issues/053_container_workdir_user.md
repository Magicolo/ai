# 053 — Container: `WORKDIR /opt/longlive` swallows output on bare `docker run`; root-owned bind-mount artifacts

- Status: open
- Severity: medium (data-loss-class CWD trap + host can't clean/inspect outputs)
- Area: containers/permissions — `worker/Dockerfile.video:97`, scripts, `output/`
- Rank rationale: AGENTS history confirms this class already bit once; observed
  `final.mp4` mode 600 uid 0 on host.

## Technical description

Video image `WORKDIR /opt/longlive`; only `run.sh` overrides with `-w /app`
(`run.sh:70-71`). `qualify.sh:31`'s summary step, ad-hoc
`docker run voyage-video ...`, and any future script forgetting `-w` land CWD in
`/opt/longlive` — relative `output/...` writes go into the ephemeral image
layer, then "missing segments" circuit-breaker fires.

Separately, containers run as root with bind-mounts `$PWD:/app`, `/tmp:/tmp`,
`$models:/models` — every render leaves root-owned files on the host:

```
$ ls -lan Voyage/output/causvid-e2e/   # observed 2026-09-25 (sweep)
drwxr-xr-x  7    0    0 4096 ... causvid-e2e        # root-owned
-rw-------  1    0    0 1251087 ... final.mp4       # mode 600, uid 0
-rw-------  1    0    0     430 ... state.json
```

Plain `rm -rf` fails (must `docker run --rm -v … rm -rf`). `PYTHONDONTWRITEBYTECODE=1`
covers `__pycache__` only, not artifacts.

## Why this is an issue

Both halves already bit real runs: a forgotten `-w` silently writes outputs
into the ephemeral image layer (the "missing segments" circuit-breaker fires
with no hint the files went to `/opt/longlive`), and root-owned `final.mp4`
(mode 600, uid 0) blocks host-side inspection, cleanup, and any non-root
tooling. For a pipeline whose whole purpose is producing host-visible video
files, output ownership is user-facing correctness, not hygiene. Fix is two
small Dockerfile/`run.sh` changes (`WORKDIR /app`, `--user`) plus a
documented cleanup recipe.

## Evidence

```
$ rg -n "^WORKDIR|-w /app" Voyage/Dockerfile Voyage/worker/Dockerfile.* Voyage/scripts/*.sh
Dockerfile:8:WORKDIR /app
worker/Dockerfile.director:38:WORKDIR /app
worker/Dockerfile.video:97:WORKDIR /opt/longlive
scripts/run.sh:70:... -w /app "${gpu_args[@]}" ...
$ sed -n '28,32p' Voyage/scripts/qualify.sh   # summary step: docker run WITHOUT -w
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/app" voyage:latest \
  python -c "import json; from tests.test_qualification import summarize_run; ..."
```

`rg -n "WORKDIR|-w /app" Voyage/Dockerfile Voyage/worker/Dockerfile.*
Voyage/scripts/*.sh`; `ls -lan` above.

## Reproduction

Bare `docker run voyage-video ...` with a relative output path; `ls -lan
Voyage/output/` after any container render.

## Source references

- `Voyage/worker/Dockerfile.video:97`; `Voyage/scripts/run.sh:70-71`,
  `qualify.sh:31`.

## Resolution candidates

1. `WORKDIR /app` in the video image; satisfy the LongLive CWD contract inside
   the worker (symlink `/opt/longlive/wan_models` or chdir in code —
   `video_causvid.py:92-98` already documents the pattern); keep `-w /app` as
   belt-and-braces.
2. `docker run --user $(id -u):$(id -g)` in `run.sh` (+ friends) with a `/models`
   writability fallback; or `umask 022` + group-writable outputs; document the
   `docker rm` recipe next to `run.sh` (currently only in AGENTS.md).

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; ownership observed live.
- 2026-09-25 (repair): re-verified refs live (`Dockerfile:8` + director `:38`
  already `/app`; video `:97` still `/opt/longlive`; `run.sh:70` still the
  only `-w /app`; `qualify.sh:31` summary `docker run` still without `-w` —
  outputs pasted in Evidence). Added `## Why this is an issue`. No staleness.
- Open: WORKDIR fix + `--user` rollout.
