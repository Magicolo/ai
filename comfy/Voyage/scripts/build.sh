#!/usr/bin/env bash
# Build the self-contained voyage image and run quality gates inside it.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t voyage:latest .
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 voyage:latest bash -c "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"
