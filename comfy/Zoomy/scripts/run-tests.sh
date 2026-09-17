#!/bin/bash
# Run the zoomy test suite inside the GPU container.
#
# Host files are fixed through the ./Zoomy bind mount (image files are
# throwaway copies), so this runs the tree as-is with no rebuild. Extra
# arguments pass through to pytest; test paths are container-relative:
#
#   ./scripts/run-tests.sh -q tests/test_rendering.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker compose --file "$ROOT/docker-compose.yml" run --rm --no-deps \
    -v "$ROOT/Zoomy:/workspace" \
    -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis \
    zoomy sh -c 'cd /workspace && python -m pytest -p no:cacheprovider "$@"' sh "$@"
