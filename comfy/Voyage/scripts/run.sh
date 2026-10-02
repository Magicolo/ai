#!/usr/bin/env bash
# Run the voyage CLI inside a container: ./scripts/run.sh <voyage args...>
#
# Env:
#   VOYAGE_IMAGE  container image (default voyage:latest, the slim CPU image;
#                 auto-selected to voyage-video:latest when a CUDA backend
#                 -- ltxv, acestep, mmaudio -- is requested, and to
#                 voyage-ltx:latest when an LTX ComfyUI backend -- ltx25,
#                 ltx23 -- is requested, unless set;
#                 bare `run.sh` (the launcher TUI, backend picked
#                 interactively) also defaults to voyage-video when the host
#                 has a GPU, so ltx25 works with no explicit variables)
#   VOYAGE_GPUS   set to 1 to pass --gpus all (auto-enabled for CUDA backends
#                 and GPU-box bare launches unless set; needed for
#                 ltxv/acestep/mmaudio)
#   VOYAGE_DRY_RUN  set to 1 to print the resolved image/gpu selection and
#                 exit (test seam; never runs docker)
#   VOYAGE_MODELS host models dir mounted at /models (default ~/.cache/voyage-models)
set -euo pipefail
cd "$(dirname "$0")/.."

# Backend-aware defaults: the CUDA worker stacks (torch + LTXV/ACE)
# only exist in voyage-video. Detect the requested backend from
# --backend <name> / --backend=<name> (defaulting to ltx25 for `generate`);
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
  requested_backend="ltx25"
fi
# Section-aware TOML sniff via the stdlib parser: reads the video/audio/sfx
# backends only, so [director] backends (or indentation/layout changes) can
# never select the wrong image. Unparseable/missing key -> empty (slim
# default), never a launcher failure. Any CUDA backend in any of the three
# sections selects the video image (issue 090: [audio].backend=acestep +
# video=fake used to stay slim, then died late in cli._require_cuda_stack;
# the same holds for [sfx].backend=mmaudio + fake/fake).
if [ -z "${requested_backend:-}" ] && [ -n "${run_dir:-}" ] \
    && [ -f "$run_dir/voyage.toml" ]; then
  requested_backend="$(RUN_DIR="$run_dir" python3 -c \
    'import os, tomllib; cfg = tomllib.load(open(os.path.join(os.environ["RUN_DIR"], "voyage.toml"), "rb")); bs = [cfg.get(s, {}).get("backend", "") for s in ("video", "audio", "sfx")]; cuda = {"ltxv", "causvid", "acestep", "mmaudio", "ltx25", "ltx23"}; print(next((b for b in bs if b in cuda), bs[0] if bs else ""))' \
    2>/dev/null || true)"
fi
needs_cuda=0
# ltx25/ltx23 run the ComfyUI worker stack, which lives in voyage-ltx
# (separate image — the validated torch 2.14/cu130 + pinned ComfyUI tree
# would break the voyage-video pins), not voyage-video.
ltx_backend=0
case "${requested_backend:-}" in
  ltxv|causvid|acestep|mmaudio) needs_cuda=1 ;;
  ltx25|ltx23) needs_cuda=1; ltx_backend=1 ;;
esac
# Bare launcher TUI: the backend is picked interactively inside the TUI, so
# no CLI signal exists. On a GPU box assume the CUDA stack so bare `run.sh`
# can generate with ltx25 and no explicit variables; explicit
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
elif [ "$ltx_backend" = "1" ]; then
  image="voyage-ltx:latest"
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
# Direct entrypoint (CUDA-banner suppression): the voyage-video stack
# inherits the nvidia/cuda entrypoint (/opt/nvidia/nvidia_entrypoint.sh),
# which prints a large CUDA banner + license block on every run. Voyage has
# its own GPU checks (torch/doctor), so exec the CLI directly. The slim
# image defines no entrypoint, making this override equivalent there.
# Single-sourced so the dry-run seam and `docker run` cannot drift apart.
entrypoint="voyage"
if [ "${VOYAGE_DRY_RUN:-}" = "1" ]; then
  printf 'image=%s\ngpus=%s\nuser=%s\nentrypoint=%s\n' "$image" "${gpu_args[*]:-none}" "${user_args[*]}" "$entrypoint"
  exit 0
fi
docker run --rm --entrypoint "$entrypoint" -e PYTHONDONTWRITEBYTECODE=1 -w /app "${user_args[@]}" "${gpu_args[@]}" "${tty_args[@]}" \
  -v "$PWD:/app" -v /tmp:/tmp -v "$models:/models" \
  "$image" "$@"
