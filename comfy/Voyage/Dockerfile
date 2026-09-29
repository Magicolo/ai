FROM python:3.12-slim

# Self-contained: no host dependencies. ffmpeg for media, pip for install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY voyage/ ./voyage/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e ".[dev]"

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
