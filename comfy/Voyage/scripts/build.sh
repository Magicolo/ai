#!/usr/bin/env bash
# Build the self-contained voyage image and run quality gates inside it.
# Scope contract (issue 092): this script gates the BAKED SNAPSHOT (no bind
# mount — what ships). scripts/gates.sh gates the LIVE TREE instead (what
# you just edited). Both rebuild first; on disagreement check Dockerfile
# COPY coverage — the file is likely missing from the image.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$SCRIPT_DIR/.."
voyage_build_image voyage:latest Dockerfile
# Same cache-hygiene contract as gates.sh: caches stay out of the tree in
# container-local /tmp (the baked /app is read-only for the runtime user).
docker run --rm "$(voyage_user_args)" "${VOYAGE_CACHE_ENV[@]}" \
  voyage:latest bash -c "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"
