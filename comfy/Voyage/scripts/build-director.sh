#!/usr/bin/env bash
# Build the Phase 3 director image (CPU torch + transformers +
# sentence-transformers) and run quality gates inside it.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage-director:latest -f worker/Dockerfile.director .
# Bind-mount the live tree (issue 054 keeps tests/ out of the image, so
# gates without the mount find no testpaths and exit 5) — same contract
# as gates.sh: the image provides the stack, the mount provides the code.
docker run --rm --user="$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  -v "$PWD:/app" \
  voyage-director:latest \
  bash -c "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"
