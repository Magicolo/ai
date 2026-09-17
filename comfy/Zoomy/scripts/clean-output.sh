#!/bin/bash
# Delete zoomy output artifacts from inside the container.
#
# Usage:
#   ./Zoomy/scripts/clean-output.sh <name-or-pattern> [...]
#
# Each argument names a file, directory, or glob relative to /output (the
# ./Zoomy/output bind mount). Globs expand inside the container, so a
# hand-typed pattern can never reach the host tree — and the user's
# ComfyUI files (Ernie_Zoom_*/Z_Zoom_*) live under Comfy/output, a
# different mount the container never sees here. Refuses empty runs,
# absolute paths, parent-directory escapes, and .gitkeep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <name-or-pattern> [...]" >&2
    exit 2
fi
for target in "$@"; do
    case "$target" in
        "" | *".."* | "/"* | ".gitkeep" | ".gitkeep/"*)
            echo "refusing to delete: $target" >&2
            exit 1
            ;;
    esac
done
# Unquoted $target below is deliberate: patterns must glob-expand inside
# the container's /output (quoting would delete one literal file name).
docker compose --file "$ROOT/docker-compose.yml" run --rm --no-deps zoomy sh -c \
    'cd /output && for target in "$@"; do rm -rf -- $target; done' sh "$@"
