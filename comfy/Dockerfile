FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt update && \
    apt upgrade --yes && \
    apt install --no-install-recommends --yes git git-lfs curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    apt autoremove --yes && \
    apt clean --yes && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --upgrade uv pip wheel setuptools
RUN uv pip install \
    transformers accelerate safetensors sentencepiece einops gradio bitsandbytes optimum imageio-ffmpeg xformers tensorrt-cu12 flash-attn onnxruntime-gpu \
    --system \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --requirement https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt \
    --requirement https://raw.githubusercontent.com/Fannovel16/comfyui_controlnet_aux/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/cubiq/ComfyUI_essentials/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/Fannovel16/ComfyUI-Frame-Interpolation/main/requirements-with-cupy.txt \
    --requirement https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Pack/Main/requirements.txt \
    --requirement https://raw.githubusercontent.com/ltdrdata/ComfyUI-Inspire-Pack/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/stavsap/comfyui-ollama/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/kijai/ComfyUI-SUPIR/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/Kosinkadink/ComfyUI-VideoHelperSuite/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/rgthree/rgthree-comfy/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/requirements.txt \
    --requirement https://raw.githubusercontent.com/kijai/ComfyUI-Florence2/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/pydn/ComfyUI-to-Python-Extension/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/DoctorDiffusion/ComfyUI-MediaMixer/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/wallish77/wlsh_nodes/main/requirements.txt \
    --requirement https://raw.githubusercontent.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt/master/requirements.txt