#!/usr/bin/env bash
# Shared constants for Voyage scripts (issue 090).
#
# Callers source this file AFTER computing SCRIPT_DIR, then `cd` to the
# Voyage root. Example (builtins only — `dirname(1)` is NOT on PATH in
# the fail-closed test env, see scripts/qualify.sh):
#   SCRIPT_DIR="$(cd "${0%/*}" && pwd)"
#   source "$SCRIPT_DIR/lib/common.sh"
#   cd "$SCRIPT_DIR/.."
#
# Scope contract (issue 092): snapshot-vs-tree headers stay at the call
# sites (build.sh = baked snapshot, gates.sh = live tree). This file
# owns only the duplicated preamble mechanics: cache env, user mapping,
# and image builds. Do NOT merge run.sh here (runtime selection vs
# build/gate lifecycle) — run.sh shares only CACHE_ENV/USER_ARGS.
#
# shellcheck disable=SC2034
VOYAGE_CACHE_ENV=(
  -e PYTHONDONTWRITEBYTECODE=1
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis
)
# Cache-hygiene contract (issue 042): caches stay out of the bind-mounted
# tree in container-local /tmp, otherwise container runs leave root-owned
# .ruff_cache/.mypy_cache/.hypothesis residue on the host.

voyage_user_args() {
  # Host-user mapping (issue 053 follow-up): images carry a real `voyager`
  # user whose UID/GID match the builder host, and --user pins the runtime
  # ids explicitly so even a stale image still leaves host-owned files.
  printf -- '--user=%s:%s' "$(id -u)" "$(id -g)"
}

voyage_build_image() {
  # Build a voyage image. Usage:
  #   voyage_build_image <tag> [dockerfile] [extra args...]
  # Default dockerfile is ./Dockerfile (slim CPU image).
  local tag="$1"
  local dockerfile="${2:-Dockerfile}"
  shift 2 || true
  docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
    -f "$dockerfile" -t "$tag" "$@" .
}
