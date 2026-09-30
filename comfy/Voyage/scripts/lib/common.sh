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

voyage_assert_no_cache_residue() {
  # Fail loud when gate caches leak into the bind-mounted tree (issue 089).
  # Container runs must keep caches in /tmp via VOYAGE_CACHE_ENV (or the
  # gates.sh inline -e flags); residue here is root-owned on the host and
  # churns build context. Usage: voyage_assert_no_cache_residue [root].
  local root="${1:-.}"
  local leaked=()
  local candidate
  for candidate in \
    "$root/.coverage" \
    "$root/.hypothesis" \
    "$root/.mypy_cache" \
    "$root/.ruff_cache" \
    "$root/coverage.xml"; do
    if [ -e "$candidate" ]; then
      leaked+=("$candidate")
    fi
  done
  if [ "${#leaked[@]}" -gt 0 ]; then
    echo "cache residue leaked into the tree: ${leaked[*]}" >&2
    echo "container runs must export VOYAGE_CACHE_ENV (see issue 042)" >&2
    return 1
  fi
  return 0
}
