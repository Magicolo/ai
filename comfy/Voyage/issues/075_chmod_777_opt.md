# 075 — `chmod 777 /opt/longlive` (+ stale-UID `chown -R`) in the video image

- Severity: MEDIUM
- File: `Voyage/worker/Dockerfile.video:166-168` (pre-created symlinks), `:213-214` (`chown -R …; chmod 777 …`), `:218` (`PYTHONPATH`)
- Area: containers — video image permissions

## Description

The image makes `/opt/longlive` (and the causvid `wan_models` link dir) world-writable so the host-mapped `voyager` user can satisfy upstream's CWD-relative `wan_models/` symlink contract at runtime. But `/opt/longlive` is on `PYTHONPATH` (`:200`); any process/user in the container can now plant or replace importable code (`pipeline.py`, `utils/…`) that the video worker imports. The preceding `chown -R "${UID}:${GID}"` also bakes the *build-time* host ids into layer ownership — rebuilding for a different host UID without a clean build leaves stale ownership that the `777` then papers over.

## Rationale

World-writable code directories on the import path collapse the user/root boundary the `voyager` user (issue 053) was created to enforce; least-privilege says writable-data and executable-code must not share a mode.

## Live evidence

Re-verified 2026-09-30 live:

```
Voyage/worker/Dockerfile.video:166-168:
RUN mkdir -p /opt/causvid/wan_models \
    && ln -sfn /models/Wan2.1-T2V-1.3B /opt/causvid/wan_models/Wan2.1-T2V-1.3B \
    && ln -sfn /models/wan_models /opt/longlive/wan_models
Voyage/worker/Dockerfile.video:213-214:
    chown -R "${UID}:${GID}" /opt/longlive /opt/causvid /opt/ACE-Step-1.5 /opt/mmaudio /opt/venvs; \
    chmod 777 /opt/longlive /opt/causvid/wan_models
Voyage/worker/Dockerfile.video:218:
ENV PYTHONPATH=/opt/longlive:/opt/ACE-Step-1.5:/opt/causvid:/opt/mmaudio:/app \
```

Shim comment at `voyage/workers/video_longlive.py:193-212` ("a root-owned checkout stays usable… zero writes") shows the *intent* (pre-created links) while the `777` goes further than the intent needs.

## Repro

`docker run --rm --user=999:999 voyage-video:latest touch /opt/longlive/pwned.py && python -c "import sys; sys.path.insert(0,'/opt/longlive'); import pwned"` succeeds; `ls -la /opt/longlive` shows `drwxrwxrwx`.

## Fix candidates

1. `chmod 755` on trees + pre-create *all* runtime symlinks/anchors at build (extend the `:166-168` pattern to every path the shims create).
2. Move the mutable link farm to a voyager-owned `/home/voyager/.voyage-opt` or tmpfs.
3. Runtime entrypoint (as root, then `gosu`/`su-exec` drop) that fixes ownership instead of baking UIDs into layers.

Overlaps with 090 (chmod scope also noted in the scripts/containers cleanup) — ownership stays here (image permissions).

## Resolution log 2026-09-30 (Rank-2 batch)

Re-verified live 2026-09-30 — CONFIRMED with line drift (file grew since
filing: pre-create `:166-168` → now `:177-179`; perms `:213-214` → now
`:224-225`; `PYTHONPATH :218` → now `:229`; same content). Shim pair
verified: `_enter_longlive_tree` (`voyage/workers/video_longlive.py:220-239`)
and `_enter_causvid_tree` (`voyage/workers/video_causvid.py:94-122`) both
skip-when-correct with zero writes; the fallback path needs owner-write on
`/opt/longlive` and `/opt/causvid/wan_models`. `run.sh:112` pins
`--user=$(id -u):$(id -g)` and `build-video.sh` forwards UID/GID at build,
so runtime UID == build UID in the normal flow and owner-write suffices
for the fallback. Overlap with `090_scripts_containers_cleanup.md`
confirmed present; ownership stays here per the issue split. No dead legs.

Fix (`Voyage/worker/Dockerfile.video` only): `chmod 777` → two `chmod 755`
lines (one per link farm) + least-privilege rationale comments at the
pre-create block and the RUN. The `chown -R` stays (provides the
owner-write bit); its ARG layer re-runs on UID change so rebuilds never
carry stale ownership.

Tests: 3 scan tests in `Voyage/tests/test_containers_rank2.py` (no `777`
outside comments, both `755` lines, pre-created links intact). Batch
48 passed; full suite 1388 passed, 5 skipped, 1 deselected. No image build
per directive — change is text-verifiable only.

Residuals: (1) runtime shim fallback under 755 unverified until the next
video build — run the causvid/longlive smoke in `build-video.sh` then;
(2) cross-UID runs against a stale image now fail loud (Errno 13) by
design — the rebuild-after-uid-change contract already covers it.

DESIGN proposal (text only, not implemented): root entrypoint that chowns
the two link farms to the runtime UID then drops to voyager
(`gosu`/`su-exec`) would remove the rebuild-on-uid-change requirement;
alternatively move the mutable link farm to a voyager-owned
`/home/voyager/.voyage-opt` with a compat symlink. Either needs an image
build to verify.
