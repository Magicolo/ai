#!/usr/bin/env bash
# Full gates: ruff lint + format check + mypy strict + pytest, all in-container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest . > /dev/null
docker run --rm --user="$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/app" voyage:latest bash -c \
  "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"
