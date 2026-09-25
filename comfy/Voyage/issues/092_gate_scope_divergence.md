# 092 — Gate-script verification scopes diverge; video image has zero gates

- Status: open
- Severity: low (two "green gates" can disagree; largest image unverified)
- Area: scripts/images — `scripts/build.sh:5-6`, `gates.sh:5-7`, `test.sh:5-6`,
  `build-video.sh:6`, `build-director.sh:6-8`,
  `worker/Dockerfile.video:83-90`, `worker/Dockerfile.director:20-23,31-33`
- Rank rationale: pass-2 finding; 037/069 cover quoting/sniffing — this is
  verification-scope divergence.

## Technical description

`build.sh` runs gates against the *baked snapshot* (no `-v` bind mount);
`gates.sh`/`test.sh` run against the *live tree* (bind-mounted over `/app`) — so
the two can disagree with uncommitted/un-COPY'd files. `build-video.sh` runs no
gates at all, and the video image can't run them (no pytest/mypy/ruff installed
— contrast the director image, which installs all three). `test.sh` runs pytest
only (no ruff/mypy).

## Why this is an issue

Two "green gates" can disagree: snapshot gates verify the baked image while
live-tree gates verify the working copy, so uncommitted or un-`COPY`'d files
pass one and fail the other with no signal which verdict to trust. Meanwhile
the largest image (video) has zero gates and `test.sh` skips lint/type — so
"green" means different things on different scripts, and the least-verified
image is the one that needs it most.

## Evidence

Script bodies verified live (re-run 2026-09-25):

```
$ sed -n '1,10p' Voyage/scripts/build.sh Voyage/scripts/gates.sh Voyage/scripts/test.sh Voyage/scripts/build-video.sh
build.sh:        docker build -t voyage:latest .
build.sh:        docker run --rm ... voyage:latest bash -c "ruff check . && ..."
gates.sh:        docker build -q -t voyage:latest . > /dev/null
gates.sh:        docker run --rm ... -v "$PWD:/app" voyage:latest bash -c "ruff check . && ..."
test.sh:         docker run --rm ... -v "$PWD:/app" voyage:latest python -m pytest "$@"
build-video.sh:  docker build -f worker/Dockerfile.video -t voyage-video:latest .
```

`build.sh` gates the baked snapshot (no bind mount); `gates.sh`/`test.sh`
gate the live tree (bind-mounted); `build-video.sh` runs no gates and the
video image installs no pytest/mypy/ruff (contrast the director image, which
installs all three).

## Reproduction

Side-by-side read; add a file under `Voyage/` not covered by Dockerfile `COPY`
and watch `build.sh` vs `gates.sh` disagree.

## Source references

- Files/lines above.

## Resolution candidates

Align on one gates invocation (bind-mount everywhere or snapshot everywhere),
add at least a smoke gate to `build-video.sh`, or document the differences in
the script headers.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (all script bodies + both Dockerfiles — match); pasted
  script contents above.
- Open: align scopes + video smoke gate.
