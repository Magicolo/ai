#!/bin/bash
# Full zoomy quality loop inside the GPU container: ruff fix, format, ruff,
# mypy strict, then pytest — the same gates CI would run, against the host
# tree via the ./Zoomy bind mount (image files are throwaway copies).
#
# Run after each completed task, not just at the end: a task is done only
# when these gates are green.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker compose --file "$ROOT/docker-compose.yml" run --rm --no-deps \
    -v "$ROOT/Zoomy:/workspace" \
    -v zoomy_mypy_cache:/tmp/mypy-cache \
    -e RUFF_CACHE_DIR=/tmp/ruff-cache \
    -e MYPY_CACHE_DIR=/tmp/mypy-cache \
    -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis \
    zoomy sh -c "cd /workspace && ruff check --fix zoomy tests && ruff format zoomy tests && ruff check zoomy tests && mypy zoomy tests && python -m pytest -p no:cacheprovider"
