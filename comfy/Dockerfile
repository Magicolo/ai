FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt update && \
    apt upgrade --yes && \
    apt install --no-install-recommends --yes git git-lfs curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    apt autoremove --yes && \
    apt clean --yes && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --upgrade uv pip wheel setuptools
RUN pip install \
    transformers accelerate safetensors sentencepiece einops gradio bitsandbytes optimum imageio-ffmpeg xformers tensorrt-cu12 flash-attn onnxruntime-gpu cupy-cuda12x \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --requirement https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt