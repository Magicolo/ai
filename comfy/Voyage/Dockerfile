# Base pinned by digest (issue 012): linux/amd64 manifest of python:3.12-slim
# resolved 2026-09-25 (`docker buildx imagetools inspect --raw
# python:3.12-slim`). The tag alone drifts silently (ffmpeg/libx264 encode
# behavior included); the digest freezes the whole Debian snapshot.
# Bump procedure: resolve the new amd64 digest the same way, update the
# FROM line, re-freeze ffmpeg/pip versions below + requirements.lock, note
# the bump in docs/INSTALL.md.
FROM python:3.12-slim@sha256:44ff437bba879d4941b710a369a8f19266aea34b29002807f0c487fabc9eec9b

# Self-contained: no host dependencies. ffmpeg for media, pip for install.
# Versions pinned (issue 012): trixie snapshot of 2026-09-25
# (`apt-cache policy ffmpeg` in the built image); unversioned apt drifts
# encode bytes across rebuilds while run_manifest.json cannot record it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg=7:7.1.5-0+deb13u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Reproducible install (issue 011): the lockfile carries the full
# transitive closure (frozen 2026-09-25, see requirements.lock header for
# the regen procedure); the editable install then adds only the voyage
# package itself (--no-deps so the lock stays the single version source).
# tests/ are NOT baked in (issue 054): build.sh/build-director.sh and
# gates.sh/test.sh all bind-mount the host tree, so in-image gates always
# test current sources instead of a stale copy; baking them only widened
# the prod image for no runtime benefit.
COPY pyproject.toml README.md requirements.lock ./
RUN pip install --no-cache-dir "pip==25.0.1" \
    && pip install --no-cache-dir -r requirements.lock \
    && rm -rf /var/lib/apt/lists/*
COPY voyage/ ./voyage/
RUN pip install --no-cache-dir --no-deps -e .

# Run as the invoking host user (issue 053 follow-up): containers share
# bind mounts with the host ($PWD:/app, /tmp:/tmp, $models:/models), so a
# root container leaves root-owned files behind. A real user carrying the
# host's UID/GID (build args, forwarded by scripts/*.sh from `id`;
# defaults 1000) keeps every written file host-owned. Rebuild after a
# host-uid change. Placed late so the slow pip layers above stay cached.
ARG UID=1000
ARG GID=1000
RUN set -eux; \
  getent group "${GID}" >/dev/null || groupadd --gid "${GID}" voyager; \
  getent passwd "${UID}" >/dev/null || useradd --uid "${UID}" --gid "${GID}" --no-create-home --shell /bin/bash voyager; \
  mkdir -p /home/voyager && chown "${UID}:${GID}" /home/voyager && chmod 755 /home/voyager
ENV HOME=/home/voyager
USER voyager

# Build-time doctor gate (issue 048): runs as the runtime user, proving the
# ffmpeg/disk/models facts resolve without root (HOME is user-writable for
# caches). Exit code is ffmpeg-only, so missing models warn without failing.
RUN voyage doctor

# Tests/gates run via scripts/gates.sh (not at build, to keep iteration fast).
CMD ["voyage", "doctor"]
