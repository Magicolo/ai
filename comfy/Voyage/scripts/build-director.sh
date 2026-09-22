#!/usr/bin/env bash
# Build the Phase 3 director image (CPU torch + transformers +
# sentence-transformers) and run quality gates inside it.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t voyage-director:latest -f worker/Dockerfile.director .
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 voyage-director:latest \
  bash -c "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"
