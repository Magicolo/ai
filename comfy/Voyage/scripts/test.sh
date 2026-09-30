#!/usr/bin/env bash
# Run the test suite inside the container (host stays clean).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$SCRIPT_DIR/.."
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest . > /dev/null
# GPU-marked tests never run by accident (issue 041): the bare default
# deselects `gpu` (in-tree gpu tests always skip without CUDA/weights;
# deselected so gates never pay for them — GPU legs live in
# qualify.sh/manual runs). Explicit args pass through untouched, e.g.
# `./scripts/test.sh -m gpu` runs only the GPU-marked tests on an idle GPU.
if [ $# -eq 0 ]; then
  set -- -m "not gpu"
fi
docker run --rm "$(voyage_user_args)" "${VOYAGE_CACHE_ENV[@]}" \
  -v "$PWD:/app" voyage:latest python -m pytest "$@"
