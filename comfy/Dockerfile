# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel

# Ubuntu 24.04 system Python 3.12 is PEP-668 externally-managed; this image IS
# the app environment (no OS tools depend on its pip), so allow system installs.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt update && \
    apt upgrade --yes && \
    apt install --no-install-recommends --yes git git-lfs curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    apt autoremove --yes && \
    apt clean --yes && \
    rm --recursive --force /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --no-cache-dir --upgrade uv pip wheel setuptools numpy
RUN pip install --no-cache-dir --upgrade \
    transformers accelerate safetensors sentencepiece einops gradio bitsandbytes optimum imageio-ffmpeg ninja tensorrt-cu13 onnxruntime-gpu cupy-cuda13x
RUN pip install --no-cache-dir comfy-cli comfy-mcp uvicorn opencv-python imageio-ffmpeg
# xformers: PyPI 0.0.35 ships binaries built for torch 2.10+cu128/py3.10 only
# (upstream split attention ops into the unpublished `mslk` package, so no
# version-matched wheel exists). Build from a pinned commit against this
# image's torch 2.14.1+cu132/py3.12 instead — verified live 2026-10-06.
# NOTE: _has_cpp_library is False on this build (attention lives in mslk),
# so ComfyUI's xformers_enabled() is False and native SDPA attention is used.
RUN pip install --no-cache-dir --no-build-isolation "xformers @ git+https://github.com/facebookresearch/xformers.git@a8aa655c444e75c69373f5d9fd09dc1f2f6a2bdb"
# Bake QoL custom nodes into image (rebuild-safe) for fresh clones.
# Host volume ./Comfy:/comfy overlays runtime but image retains baked code.
RUN mkdir --parents /comfy/custom_nodes && \
    if [ -d /comfy/custom_nodes/comfy-loopback-buffer ]; then git --work-tree=/comfy/custom_nodes/comfy-loopback-buffer --git-dir=/comfy/custom_nodes/comfy-loopback-buffer/.git pull --ff-only; else git clone https://github.com/holo-q/comfy-loopback-buffer /comfy/custom_nodes/comfy-loopback-buffer; fi && \
    if [ -d /comfy/custom_nodes/ComfyUI-VideoHelperSuite ]; then git --work-tree=/comfy/custom_nodes/ComfyUI-VideoHelperSuite --git-dir=/comfy/custom_nodes/ComfyUI-VideoHelperSuite/.git pull --ff-only; else git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfy/custom_nodes/ComfyUI-VideoHelperSuite; fi && \
    if [ -d /comfy/custom_nodes/cg-use-everywhere ]; then git --work-tree=/comfy/custom_nodes/cg-use-everywhere --git-dir=/comfy/custom_nodes/cg-use-everywhere/.git pull --ff-only; else git clone https://github.com/chrisgoringe/cg-use-everywhere /comfy/custom_nodes/cg-use-everywhere; fi && \
    if [ -d /comfy/custom_nodes/comfyui-workflow-prettier ]; then git --work-tree=/comfy/custom_nodes/comfyui-workflow-prettier --git-dir=/comfy/custom_nodes/comfyui-workflow-prettier/.git pull --ff-only; else git clone https://github.com/deepme987/comfyui-workflow-prettier /comfy/custom_nodes/comfyui-workflow-prettier; fi
# Requirements come from the host checkout (the tree bind-mounted at runtime) so the image
# always matches the mounted ComfyUI commit and custom-node set. A remote URL would stay
# cached while upstream moves on (that is how comfy-aimdo went missing). COPY --parents
# preserves each node's path — a plain wildcard COPY would collapse same-named files.
COPY --parents Comfy/requirements.txt Comfy/custom_nodes/*/requirements*.txt /opt/comfy-requirements/
RUN for req in /opt/comfy-requirements/Comfy/requirements.txt /opt/comfy-requirements/Comfy/custom_nodes/*/requirements*.txt /comfy/custom_nodes/*/requirements*.txt; do \
      [ -f "$req" ] || continue; \
      echo "[Dockerfile] pip install --requirement $req"; \
      pip install --no-cache-dir --requirement "$req"; \
    done
COPY entry.sh ./entry.sh
ENTRYPOINT ["./entry.sh"]
