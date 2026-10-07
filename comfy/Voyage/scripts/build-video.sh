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
# SonicMaster mastering venv (Track D): same isolated stack as the ltx
# image (finalize runs under this image too), spawned via
# VOYAGE_MASTERING_PYTHON. Import-level only (no GPU needed for imports;
# the mastering render is proven by a GPU run, not the build).
docker run --rm "$(voyage_user_args)" "${VOYAGE_CACHE_ENV[@]}" \
  --entrypoint /opt/venvs/mastering/bin/python voyage-video:latest -c "
import torch, transformers, diffusers, soundfile, safetensors, huggingface_hub
import pydantic, datasets, torchaudio
assert torch.__version__.startswith('2.4.0'), torch.__version__
assert transformers.__version__ == '4.44.0', transformers.__version__
assert diffusers.__version__ == '0.30.0', diffusers.__version__
assert datasets.__version__ == '3.6.0', datasets.__version__
assert torchaudio.__version__.startswith('2.4.0'), torchaudio.__version__
print('mastering venv ok')
"
