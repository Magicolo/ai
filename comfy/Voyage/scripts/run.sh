#!/usr/bin/env bash
# Run the voyage CLI inside a container: ./scripts/run.sh <voyage args...>
#
# Env:
#   VOYAGE_IMAGE  container image (default voyage:latest, the slim CPU image;
#                 use voyage-video:latest for the CUDA worker stack)
#   VOYAGE_GPUS   set to 1 to pass --gpus all (needed for longlive2 backend)
#   VOYAGE_MODELS host models dir mounted at /models (default ~/.cache/voyage-models)
set -euo pipefail
cd "$(dirname "$0")/.."
image="${VOYAGE_IMAGE:-voyage:latest}"
models="${VOYAGE_MODELS:-$HOME/.cache/voyage-models}"
mkdir -p "$models"
gpu_args=()
if [ "${VOYAGE_GPUS:-0}" = "1" ]; then
  gpu_args=(--gpus all)
fi
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 "${gpu_args[@]}" \
  -v "$PWD:/app" -v /tmp:/tmp -v "$models:/models" \
  "$image" voyage "$@"
