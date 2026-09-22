#!/usr/bin/env bash
# Run the voyage CLI inside the container: ./scripts/run.sh <voyage args...>
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q -t voyage:latest . > /dev/null
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/app" -v /tmp:/tmp voyage:latest voyage "$@"
