#!/usr/bin/env bash
# Run the test suite inside the container (host stays clean).
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest . > /dev/null
docker run --rm --user="$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  -v "$PWD:/app" voyage:latest python -m pytest "$@"
