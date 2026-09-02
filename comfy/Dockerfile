FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt update && \
    apt upgrade --yes && \
    apt install --no-install-recommends --yes git git-lfs curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    apt autoremove --yes && \
    apt clean --yes && \
    rm --recursive --force /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --no-cache-dir --upgrade uv pip wheel setuptools numpy
RUN pip install --no-cache-dir --upgrade \
    transformers accelerate safetensors sentencepiece einops gradio bitsandbytes optimum imageio-ffmpeg xformers tensorrt-cu12 onnxruntime-gpu cupy-cuda12x \
    --requirement https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt \
    --requirement https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main/requirements.txt
RUN pip install --no-cache-dir comfy-cli comfy-mcp uvicorn opencv-python imageio-ffmpeg
# Bake QoL custom nodes into image (rebuild-safe) and install their requirements.
# Host volume ./Comfy:/comfy overlays runtime but image retains deps + baked code for fresh clones.
# Pattern matches existing --requirement installs above.
RUN mkdir --parents /comfy/custom_nodes && \
    if [ -d /comfy/custom_nodes/comfy-loopback-buffer ]; then git --work-tree=/comfy/custom_nodes/comfy-loopback-buffer --git-dir=/comfy/custom_nodes/comfy-loopback-buffer/.git pull --ff-only; else git clone https://github.com/holo-q/comfy-loopback-buffer /comfy/custom_nodes/comfy-loopback-buffer; fi && \
    if [ -d /comfy/custom_nodes/ComfyUI-VideoHelperSuite ]; then git --work-tree=/comfy/custom_nodes/ComfyUI-VideoHelperSuite --git-dir=/comfy/custom_nodes/ComfyUI-VideoHelperSuite/.git pull --ff-only; else git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfy/custom_nodes/ComfyUI-VideoHelperSuite; fi && \
    if [ -d /comfy/custom_nodes/cg-use-everywhere ]; then git --work-tree=/comfy/custom_nodes/cg-use-everywhere --git-dir=/comfy/custom_nodes/cg-use-everywhere/.git pull --ff-only; else git clone https://github.com/chrisgoringe/cg-use-everywhere /comfy/custom_nodes/cg-use-everywhere; fi && \
    if [ -d /comfy/custom_nodes/comfyui-workflow-prettier ]; then git --work-tree=/comfy/custom_nodes/comfyui-workflow-prettier --git-dir=/comfy/custom_nodes/comfyui-workflow-prettier/.git pull --ff-only; else git clone https://github.com/deepme987/comfyui-workflow-prettier /comfy/custom_nodes/comfyui-workflow-prettier; fi && \
    touch /comfy/custom_nodes/comfy-loopback-buffer/requirements.txt /comfy/custom_nodes/cg-use-everywhere/requirements.txt /comfy/custom_nodes/comfyui-workflow-prettier/requirements.txt && \
    pip install --no-cache-dir --requirement https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt --requirement https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main/requirements.txt --requirement /comfy/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt --requirement /comfy/custom_nodes/comfy-loopback-buffer/requirements.txt --requirement /comfy/custom_nodes/cg-use-everywhere/requirements.txt --requirement /comfy/custom_nodes/comfyui-workflow-prettier/requirements.txt
COPY entry.sh ./entry.sh
ENTRYPOINT ["./entry.sh"]
