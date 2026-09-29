#!/usr/bin/env bash
# Run the test suite inside the container (host stays clean).
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest . > /dev/null
# GPU-marked tests never run by accident (issue 041): the bare default
# deselects `gpu` (no in-tree gpu tests exist yet — GPU legs live in
# qualify.sh/manual runs). Explicit args pass through untouched, e.g.
# `./scripts/test.sh -m gpu` runs only the GPU-marked tests on an idle GPU.
if [ $# -eq 0 ]; then
  set -- -m "not gpu"
fi
docker run --rm --user="$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  -v "$PWD:/app" voyage:latest python -m pytest "$@"
