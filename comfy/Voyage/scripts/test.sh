#!/usr/bin/env bash
# Run the test suite inside the container (host stays clean).
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q -t voyage:latest . > /dev/null
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/app" voyage:latest python -m pytest "$@"
