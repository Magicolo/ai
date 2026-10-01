#!/usr/bin/env bash
# Build the CUDA video worker image (long pole: torch +
# flash-attn). Run in background and poll /tmp/videobuild.log.
# Ends with the issue-092 smoke gate: both video workers must import
# and delegate main() to the shared serve map. CPU-only (no --gpus, no
# models, no network) — it catches broken stacks/pins, not numerics.
# NOTE: the CUDA base image has no `python` shim, so the smoke runs under
# --entrypoint python3 (the nvidia entrypoint execs `python` otherwise).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$SCRIPT_DIR/.."
voyage_build_image voyage-video:latest worker/Dockerfile.video
docker run --rm "$(voyage_user_args)" "${VOYAGE_CACHE_ENV[@]}" \
  --entrypoint python3 voyage-video:latest -c "
from voyage.workers import video_causvid, video_ltxv
import inspect
for mod in (video_ltxv, video_causvid):
    src = inspect.getsource(mod.main)
    assert 'standard_serve_map' in src, mod.__name__
    for handler in ('handle_init', 'handle_health', 'handle_generate_blocks', 'handle_benchmark', 'handle_evict_gpu', 'handle_rebuild', 'handle_resume'):
        assert handler in src, (mod.__name__, handler)
print('video smoke ok: 2 workers import + delegate to shared serve map')
"
