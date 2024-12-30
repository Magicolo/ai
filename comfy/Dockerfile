FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt update && \
    apt upgrade --yes && \
    apt install --yes git git-lfs curl
RUN pip install gradio xformers tensorrt flash-attn onnxruntime-gpu \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --requirement https://raw.githubusercontent.com/comfyanonymous/ComfyUI/refs/heads/master/requirements.txt \
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
    --requirement https://raw.githubusercontent.com/rgthree/rgthree-comfy/main/requirements.txt
RUN apt autoremove --yes && \
    apt clean --yes && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 