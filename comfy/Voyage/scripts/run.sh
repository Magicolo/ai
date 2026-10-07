#!/usr/bin/env bash
# Run the voyage CLI inside a container: ./scripts/run.sh <voyage args...>
#
# Env:
#   VOYAGE_IMAGE  container image (default voyage:latest, the slim CPU image;
#                 auto-selected to voyage-video:latest when a CUDA backend
#                 -- ltxv, acestep, mmaudio -- is requested, and to
#                 voyage-ltx:latest when an LTX ComfyUI backend -- ltx25,
#                 ltx23 -- is requested, and to voyage-video:latest when
#                 director.backend is llama (the sidecar binary lives only
#                 in the CUDA images), unless set;
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
# --backend <name> / --backend=<name>; for `generate <NAME>` (positional
# NAME, no flag) resolve output/<NAME>/manifest.json and read the stored
# backend, falling back to ltx25 only when no manifest is readable;
# for run-like commands with --run DIR, read it from DIR/manifest.json
# (CLI-is-config: the manifest carries the effective config; no TOML).
# Explicit VOYAGE_IMAGE / VOYAGE_GPUS always win.
#
# Argument-contract: --backend/--run/--name are each accepted in BOTH the
# separate-arg form (--run DIR) and the equals form (--run=DIR). The sniff
# loop below must keep the three flags symmetric — a dropped form leaves
# run_dir/backend unset and silently selects the slim image for a CUDA run
# (issues 037/069). --name resolves to output/<name>, mirroring the CLI
# resolver (relative to this directory, which is also the CLI cwd root).
# Covers only flag parsing; values are validated later.
requested_backend=""
run_dir=""
prev_arg=""
for arg in "$@"; do
  if [ "$prev_arg" = "--backend" ] || [ "$prev_arg" = "--run" ] || [ "$prev_arg" = "--name" ]; then
    if [ "$prev_arg" = "--backend" ]; then
      requested_backend="$arg"
    elif [ "$prev_arg" = "--run" ]; then
      run_dir="$arg"
    else
      run_dir="output/$arg"
    fi
    prev_arg=""
  elif [[ "$arg" == --backend=* ]]; then
    requested_backend="${arg#--backend=}"
  elif [[ "$arg" == --run=* ]]; then
    run_dir="${arg#--run=}"
  elif [[ "$arg" == --name=* ]]; then
    run_dir="output/${arg#--name=}"
  elif [[ "$arg" == --backend || "$arg" == --run || "$arg" == --name ]]; then
    prev_arg="$arg"
  else
    prev_arg=""
  fi
done
# `generate` takes a positional run NAME (no --run/--name flag): resolve
# output/<NAME> so the manifest sniff below selects the stored backend.
# Without this every generate fell back to ltx25 and e.g. ltxv runs
# landed in voyage-ltx, which lacks the ltx_video module (WORKER_ERROR
# at worker start). Skips generate's value-flag arguments (--segments N,
# --duration LEN) so a value is never mistaken for the NAME.
if [ "${1:-}" = "generate" ] && [ "$#" -ge 2 ] \
    && [ -z "${run_dir:-}" ] && [ -z "${requested_backend:-}" ]; then
  skip_next=0
  for arg in "${@:2}"; do
    if [ "$skip_next" = "1" ]; then skip_next=0; continue; fi
    case "$arg" in
      --segments|--duration) skip_next=1 ;;
      --segments=*|--duration=*|--verbose|--no-color|--quiet) ;;
      -*) ;;
      *) run_dir="output/$arg"; break ;;
    esac
  done
  unset skip_next
fi
# Manifest sniff via the stdlib JSON parser: reads the video/audio/sfx
# backends from the flat manifest root (the effective config IS the
# manifest root — segments/finalize policy ride alongside and are
# ignored here), plus director.backend: the default director is `llama`
# (loopback llama-server sidecar, a GPU workload baked only into the
# CUDA images — never the slim image), so video=fake + director=llama
# must select a CUDA image (issue 205; the old comment here claimed
# [director] backends could never select the wrong image — that
# overclaimed: fake+llama sniffed `fake` and landed in slim, dying late
# with `cannot spawn llama-server ... Errno 2`).
# Non-dict sections / non-string backends warn on stderr and are ignored
# (issue 240: the old comprehension raised AttributeError, hidden by
# 2>/dev/null into a silent wrong-image pick — stderr now stays loud).
# An unreadable manifest or non-dict root warns and yields no signal:
# generate keeps the historical ltx25 fallback below, while other verbs
# exit 2 naming the manifest (fail-closed, like the qualify.sh gates —
# the CLI would fail loud on the same file anyway).
# Any CUDA backend in any of the three media sections selects the
# video image (issue 090: [audio].backend=acestep + video=fake used to
# stay slim, then died late in cli._require_cuda_stack; the same holds
# for [sfx].backend=mmaudio + fake/fake).
if [ -z "${requested_backend:-}" ] && [ -n "${run_dir:-}" ] \
    && [ -f "$run_dir/manifest.json" ]; then
  if sniff_output="$(RUN_DIR="$run_dir" python3 -c '
import json, os, sys
path = os.path.join(os.environ["RUN_DIR"], "manifest.json")
try:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
except (OSError, ValueError) as exc:
    print(f"run.sh: warning: cannot parse {path}: {exc}; using default image selection", file=sys.stderr)
    sys.exit(3)
if not isinstance(config, dict):
    print(f"run.sh: warning: {path} root is not an object; using default image selection", file=sys.stderr)
    sys.exit(3)
def section_backend(name):
    section = config.get(name, {})
    if not isinstance(section, dict):
        print(f"run.sh: warning: {path} section [{name}] is not an object; ignoring it", file=sys.stderr)
        return ""
    backend = section.get("backend", "")
    if not isinstance(backend, str):
        print(f"run.sh: warning: {path} section [{name}] backend is not a string; ignoring it", file=sys.stderr)
        return ""
    return backend
video = section_backend("video")
audio = section_backend("audio")
sfx = section_backend("sfx")
director = section_backend("director")
cuda = {"ltxv", "causvid", "acestep", "mmaudio", "ltx25", "ltx23"}
for backend in (video, audio, sfx):
    if backend in cuda:
        print(backend)
        break
else:
    print("llama" if director == "llama" else video)
')"; then
    requested_backend="$sniff_output"
  elif [ "${1:-}" = "generate" ]; then
    # Malformed/unreadable manifest (already warned on stderr above):
    # keep the historical ltx25 fallback below, never a launcher failure.
    requested_backend=""
  else
    echo "run.sh: error: cannot sniff a backend from $run_dir/manifest.json; refusing to guess an image" >&2
    exit 2
  fi
fi
# No signal at all (bare `generate` with no NAME, or a NAME whose manifest
# is missing/unreadable): historical default is the LTX stack.
if [ -z "${requested_backend:-}" ] && [ "${1:-}" = "generate" ]; then
  requested_backend="ltx25"
fi
needs_cuda=0
# ltx25/ltx23 run the ComfyUI worker stack, which lives in voyage-ltx
# (separate image — the validated torch 2.14/cu130 + pinned ComfyUI tree
# would break the voyage-video pins), not voyage-video.
ltx_backend=0
case "${requested_backend:-}" in
  ltxv|causvid|acestep|mmaudio) needs_cuda=1 ;;
  ltx25|ltx23) needs_cuda=1; ltx_backend=1 ;;
  # Issue 205: the llama director sidecar is a GPU workload baked only
  # into the CUDA images, so it needs a CUDA image even when every media
  # backend is fake. voyage-video (not -ltx) is its home: both CUDA
  # images carry the binary, but -ltx is reserved for the ComfyUI worker
  # stack a fake-video run never needs.
  llama) needs_cuda=1 ;;
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
# LTX worker tuning passthrough (Track C strength/carry matrix): the
# video workers read VOYAGE_LTX_STRENGTH / VOYAGE_LTX_CARRY / VOYAGE_LTX_FAST
# from their own environment (fail-loud validators in video_ltx25.py), but
# `docker run` below never forwarded them — so `VOYAGE_LTX_STRENGTH=0.8
# ./scripts/run.sh generate NAME` silently rendered at strength 1.0.
# Forward each only when set in the caller environment (empty `-e` would
# inject empty strings the validators reject).
ltx_env_args=()
for ltx_env_name in VOYAGE_LTX_STRENGTH VOYAGE_LTX_CARRY VOYAGE_LTX_FAST; do
  if [ -n "${!ltx_env_name:-}" ]; then
    ltx_env_args+=(-e "${ltx_env_name}=${!ltx_env_name}")
  fi
done
# Single-sourced so the dry-run seam and `docker run` cannot drift apart.
entrypoint="voyage"
if [ "${VOYAGE_DRY_RUN:-}" = "1" ]; then
  printf 'image=%s\ngpus=%s\nuser=%s\nentrypoint=%s\nenv=%s\n' "$image" "${gpu_args[*]:-none}" "${user_args[*]}" "$entrypoint" "${ltx_env_args[*]:-none}"
  exit 0
fi
docker run --rm --entrypoint "$entrypoint" -e PYTHONDONTWRITEBYTECODE=1 "${ltx_env_args[@]}" -w /app "${user_args[@]}" "${gpu_args[@]}" "${tty_args[@]}" \
  -v "$PWD:/app" -v /tmp:/tmp -v "$models:/models" \
  "$image" "$@"
