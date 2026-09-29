#!/usr/bin/env bash
# Run the voyage CLI inside a container: ./scripts/run.sh <voyage args...>
#
# Env:
#   VOYAGE_IMAGE  container image (default voyage:latest, the slim CPU image;
#                 auto-selected to voyage-video:latest when a CUDA backend
#                 -- ltxv, longlive2, acestep -- is requested, unless set;
#                 bare `run.sh` (the launcher TUI, backend picked
#                 interactively) also defaults to voyage-video when the host
#                 has a GPU, so ltxv works with no explicit variables)
#   VOYAGE_GPUS   set to 1 to pass --gpus all (auto-enabled for CUDA backends
#                 and GPU-box bare launches unless set; needed for
#                 longlive2/ltxv/acestep)
#   VOYAGE_DRY_RUN  set to 1 to print the resolved image/gpu selection and
#                 exit (test seam; never runs docker)
#   VOYAGE_MODELS host models dir mounted at /models (default ~/.cache/voyage-models)
set -euo pipefail
cd "$(dirname "$0")/.."

# Backend-aware defaults: the CUDA worker stacks (torch + LongLive/LTXV/ACE)
# only exist in voyage-video. Detect the requested backend from
# --backend <name> / --backend=<name> (defaulting to ltxv for `generate`);
# for run-like commands with --run DIR, read it from DIR/voyage.toml.
# Explicit VOYAGE_IMAGE / VOYAGE_GPUS always win.
#
# Argument-contract: --backend and --run are each accepted in BOTH the
# separate-arg form (--run DIR) and the equals form (--run=DIR). The sniff
# loop below must keep the two flags symmetric — a dropped form leaves
# run_dir/backend unset and silently selects the slim image for a CUDA run
# (issues 037/069). Covers only flag parsing; values are validated later.
requested_backend=""
run_dir=""
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
  elif [[ "$arg" == --run=* ]]; then
    run_dir="${arg#--run=}"
  elif [[ "$arg" == --backend || "$arg" == --run ]]; then
    prev_arg="$arg"
  else
    prev_arg=""
  fi
done
if [ -z "${requested_backend:-}" ] && [ "${1:-}" = "generate" ]; then
  requested_backend="ltxv"
fi
# Section-aware TOML sniff via the stdlib parser: reads the [video] backend
# only, so [audio]/[director] backends (or indentation/layout changes) can
# never select the wrong image. Unparseable/missing key -> empty (slim
# default), never a launcher failure.
if [ -z "${requested_backend:-}" ] && [ -n "${run_dir:-}" ] \
    && [ -f "$run_dir/voyage.toml" ]; then
  requested_backend="$(RUN_DIR="$run_dir" python3 -c \
    'import os, tomllib; print(tomllib.load(open(os.path.join(os.environ["RUN_DIR"], "voyage.toml"), "rb")).get("video", {}).get("backend", ""))' \
    2>/dev/null || true)"
fi
needs_cuda=0
case "${requested_backend:-}" in
  ltxv|longlive2|causvid|acestep) needs_cuda=1 ;;
esac
# Bare launcher TUI: the backend is picked interactively inside the TUI, so
# no CLI signal exists. On a GPU box assume the CUDA stack so bare `run.sh`
# can generate with ltxv/longlive2 and no explicit variables; explicit
# VOYAGE_IMAGE / VOYAGE_GPUS always win. On GPU-less boxes stay slim (the
# TUI still runs fake-backend smoke runs; picking a CUDA backend there
# fast-fails to the form with the relaunch hint). The probe is a cheap
# `nvidia-smi -L` and never fails the script.
host_has_gpu=0
if [ "$#" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi -L >/dev/null 2>&1; then
  host_has_gpu=1
fi
if [ "$needs_cuda" = "1" ] || [ "$host_has_gpu" = "1" ]; then
  want_cuda=1
else
  want_cuda=0
fi
if [ -n "${VOYAGE_IMAGE:-}" ]; then
  image="$VOYAGE_IMAGE"
elif [ "$want_cuda" = "1" ]; then
  image="voyage-video:latest"
else
  image="voyage:latest"
fi
models="${VOYAGE_MODELS:-$HOME/.cache/voyage-models}"
mkdir -p "$models"
gpu_args=()
if [ "${VOYAGE_GPUS:-}" = "1" ]; then
  gpu_args=(--gpus all)
elif [ -z "${VOYAGE_GPUS:-}" ] && [ "$want_cuda" = "1" ]; then
  gpu_args=(--gpus all)
fi
# The bare-command launcher TUI needs a real terminal: allocate one when
# attached, stay pipe-friendly otherwise.
tty_args=()
if [ -t 0 ] && [ -t 1 ]; then
  tty_args=(--interactive --tty)
fi
# Host-user mapping (issue 053 follow-up): images carry a real `voyager`
# user whose UID/GID match the builder host, and --user pins the runtime
# ids explicitly so even a stale image (built under other ids) still
# leaves host-owned files on the bind mounts ($PWD:/app, /tmp:/tmp,
# $models:/models) instead of root-owned ones.
user_args=("--user=$(id -u):$(id -g)")
if [ "${VOYAGE_DRY_RUN:-}" = "1" ]; then
  printf 'image=%s\ngpus=%s\nuser=%s\n' "$image" "${gpu_args[*]:-none}" "${user_args[*]}"
  exit 0
fi
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -w /app "${user_args[@]}" "${gpu_args[@]}" "${tty_args[@]}" \
  -v "$PWD:/app" -v /tmp:/tmp -v "$models:/models" \
  "$image" voyage "$@"
