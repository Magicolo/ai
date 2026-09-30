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
