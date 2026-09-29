#!/usr/bin/env bash
# Build the self-contained voyage image and run quality gates inside it.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest .
# Same cache-hygiene contract as gates.sh: caches stay out of the tree in
# container-local /tmp (the baked /app is read-only for the runtime user).
docker run --rm --user="$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  voyage:latest bash -c "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"
