#!/usr/bin/env bash
# Build the LTX worker image (long pole: torch 2.14/cu130 + ComfyUI stack).
# Run in background and poll /tmp/ltxbuild.log.
# Ends with the smoke gate: pinned ComfyUI + patched ComfyUI-GGUF must
# import in-process (the Phase-2 execution vehicle) and the GGUF node map
# must carry the loaders both families need. Needs --gpus: importing
# comfy.model_management probes torch.cuda at module scope, so CPU-only
# docker cannot even import the stack (RuntimeError: no NVIDIA driver).
# NOTE: the CUDA base image has no `python` shim, so the smoke runs under
# --entrypoint python3 (the nvidia entrypoint execs `python` otherwise).
# Phase 2 extends this gate to the video_ltx25/video_ltx23 workers
# (issue-092-style: import + delegate to the shared serve map).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$SCRIPT_DIR/.."
# Stage the LTX prefix-freeze pack for the image bake: it lives in the
# Comfy checkout (outside this Voyage build context), so copy it under
# worker/ where the Dockerfile COPY can reach it. Removed when the build
# finishes, pass or fail; gitignored build debris, never committed.
PACK_SRC="$SCRIPT_DIR/../../Comfy/custom_nodes/ltx_mask_utils"
[ -d "$PACK_SRC" ] || { echo "missing LTX mask pack: $PACK_SRC" >&2; exit 1; }
rm -rf worker/ltx_mask_utils
cp -r "$PACK_SRC" worker/ltx_mask_utils
trap 'rm -rf worker/ltx_mask_utils' EXIT
voyage_build_image voyage-ltx:latest worker/Dockerfile.ltx
# llama-server sidecar binary (DESIGN §140 llama entry): the default generate
# path (video=ltx25 + director=llama) spawns it, so the bake must carry it.
# File-existence only (no --version: the CUDA binary cannot start without
# libcuda.so.1, absent in the build container without --gpus).
docker run --rm --entrypoint test voyage-ltx:latest -x /opt/llama.cpp/bin/llama-server
docker run --rm --entrypoint test voyage-ltx:latest -x /usr/local/bin/llama-server
docker run --rm --gpus all "$(voyage_user_args)" "${VOYAGE_CACHE_ENV[@]}" \
  --entrypoint python3 voyage-ltx:latest -c "
import importlib.util, sys, types
import torch
import execution, comfy.model_management, comfy.sample
import gguf

# Fake-server surface the in-process executor needs (execution.py):
# client_id None = silent, send_sync + queue_updated no-ops.
class _FakeServer:
    client_id = None
    def send_sync(self, *a, **k): pass
    def queue_updated(self): pass
ex = execution.PromptExecutor(_FakeServer())
assert ex.server.client_id is None

# ComfyUI-GGUF nodes carry relative imports (from .loader import ...),
# so they load as a package anchored at the clone dir (same layout as
# the validated experiment tree: ComfyUI/custom_nodes/ComfyUI-GGUF).
gguf_dir = '/opt/comfyui/custom_nodes/ComfyUI-GGUF'
pkg = types.ModuleType('gguf_pkg'); pkg.__path__ = [gguf_dir]
sys.modules['gguf_pkg'] = pkg
spec = importlib.util.spec_from_file_location('gguf_pkg.nodes', gguf_dir + '/nodes.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['gguf_pkg.nodes'] = mod
spec.loader.exec_module(mod)
for name in ('UnetLoaderGGUF', 'CLIPLoaderGGUF', 'DualCLIPLoaderGGUF'):
    assert name in mod.NODE_CLASS_MAPPINGS, name

# Gemma4 patch marker (build asserts it too; re-asserted at runtime).
src = open(gguf_dir + '/loader.py').read()
assert '\"gemma4\"' in src and 'LTXV_BF16_PARAMETERS' in src
# LTX prefix-freeze pack (baked by the Dockerfile COPY above; torch-only).
mask_dir = '/opt/comfyui/custom_nodes/ltx_mask_utils'
spec = importlib.util.spec_from_file_location('ltx_mask_utils.prefix_freeze', mask_dir + '/prefix_freeze.py')
maskmod = importlib.util.module_from_spec(spec)
sys.modules['ltx_mask_utils.prefix_freeze'] = maskmod
spec.loader.exec_module(maskmod)
assert 'LTXPrefixFreeze' in maskmod.NODE_CLASS_MAPPINGS, 'LTXPrefixFreeze'

print('ltx smoke ok: torch', torch.__version__, '+ comfy exec + GGUF loaders + gemma4 patch')
"
# ACE-Step music venv (DESIGN §140 audio continuity): the ACE stack is
# isolated from the validated LTX freeze (transformers pin), so the audio
# worker spawns with this interpreter via VOYAGE_ACESTEP_PYTHON.
# Import-level only (no GPU needed for imports; the take render is proven
# by a GPU run, not the build).
docker run --rm --entrypoint /opt/venvs/acestep/bin/python voyage-ltx:latest -c "
import acestep.handler, acestep.llm_inference, acestep.inference
assert hasattr(acestep.handler, 'AceStepHandler'), 'AceStepHandler'
assert hasattr(acestep.llm_inference, 'LLMHandler'), 'LLMHandler'
assert hasattr(acestep.inference, 'generate_music'), 'generate_music'
print('ace venv ok')
"
# MMAudio SFX venv (DESIGN §140 SFX continuity): the SFX stack is isolated
# from the validated LTX freeze, so finalize spawns the SFX worker with
# this interpreter via VOYAGE_SFX_PYTHON.
# Import-level only (no GPU needed for imports; the SFX bed render is
# proven by a GPU run, not the build).
docker run --rm --entrypoint /opt/venvs/sfx/bin/python voyage-ltx:latest -c "
import mmaudio.ext.autoencoder.autoencoder, mmaudio.model.utils.features_utils
import mmaudio.model.networks
from mmaudio.model.flow_matching import FlowMatching
import av
assert FlowMatching is not None, 'FlowMatching'
print('sfx venv ok')
"
