#!/usr/bin/env bash
# Run the voyage CLI inside a container: ./scripts/run.sh <voyage args...>
#
# Env:
#   VOYAGE_IMAGE  container image (default voyage:latest, the slim CPU image;
#                 auto-selected to voyage-video:latest when a CUDA backend
#                 -- ltxv, longlive2, acestep -- is requested, unless set)
#   VOYAGE_GPUS   set to 1 to pass --gpus all (auto-enabled for CUDA backends
#                 unless set; needed for longlive2/ltxv/acestep)
#   VOYAGE_MODELS host models dir mounted at /models (default ~/.cache/voyage-models)
set -euo pipefail
cd "$(dirname "$0")/.."

# Backend-aware defaults: the CUDA worker stacks (torch + LongLive/LTXV/ACE)
# only exist in voyage-video. Detect the requested backend from
# --backend <name> / --backend=<name> (defaulting to ltxv for `generate`);
# for run-like commands with --run DIR, read it from DIR/voyage.toml.
# Explicit VOYAGE_IMAGE / VOYAGE_GPUS always win.
requested_backend=""
prev_arg=""
for arg in "$@"; do
  if [ "$prev_arg" = "--backend" ] || [ "$prev_arg" = "--run" ]; then
    if [ "$prev_arg" = "--backend" ]; then
      requested_backend="$arg"
    else
      run_dir="$arg"
    fi
    prev_arg=""
  elif [[ "$arg" == --backend=* ]]; then
    requested_backend="${arg#--backend=}"
  elif [[ "$arg" == --backend || "$arg" == --run ]]; then
    prev_arg="$arg"
  else
    prev_arg=""
  fi
done
if [ -z "${requested_backend:-}" ] && [ "${1:-}" = "generate" ]; then
  requested_backend="ltxv"
fi
if [ -z "${requested_backend:-}" ] && [ -n "${run_dir:-}" ] \
    && [ -f "$run_dir/voyage.toml" ]; then
  requested_backend="$(grep -E '^backend *= *"' "$run_dir/voyage.toml" \
    | head -n 1 | sed -E 's/.*"(.*)".*/\1/')"
fi
needs_cuda=0
case "${requested_backend:-}" in
  ltxv|longlive2|acestep) needs_cuda=1 ;;
esac
if [ -n "${VOYAGE_IMAGE:-}" ]; then
  image="$VOYAGE_IMAGE"
elif [ "$needs_cuda" = "1" ]; then
  image="voyage-video:latest"
else
  image="voyage:latest"
fi
models="${VOYAGE_MODELS:-$HOME/.cache/voyage-models}"
mkdir -p "$models"
gpu_args=()
if [ "${VOYAGE_GPUS:-}" = "1" ]; then
  gpu_args=(--gpus all)
elif [ -z "${VOYAGE_GPUS:-}" ] && [ "$needs_cuda" = "1" ]; then
  gpu_args=(--gpus all)
fi
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -w /app "${gpu_args[@]}" \
  -v "$PWD:/app" -v /tmp:/tmp -v "$models:/models" \
  "$image" voyage "$@"
