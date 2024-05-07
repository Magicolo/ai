FROM python

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl ffmpeg melt gmic
RUN pip install --upgrade pip opencv-python torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN git clone --recursive https://github.com/comfyanonymous/ComfyUI.git /comfy
RUN pip install --requirement comfy/requirements.txt
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/ltdrdata/ComfyUI-Manager.git
RUN pip install --requirement comfy/custom_nodes/ComfyUI-Manager/requirements.txt
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
RUN pip install --requirement comfy/custom_nodes/ComfyUI-Impact-Pack/requirements.txt
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git
RUN pip install --requirement comfy/custom_nodes/ComfyUI-Inspire-Pack/requirements.txt
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
RUN pip install --requirement comfy/custom_nodes/ComfyUI-Advanced-ControlNet/requirements.txt
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/Fannovel16/comfyui_controlnet_aux.git
RUN pip install --requirement comfy/custom_nodes/comfyui_controlnet_aux/requirements.txt
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
HEALTHCHECK --interval=5s --timeout=30s --start-period=5s --retries=10 CMD curl http://localhost:8188
ENTRYPOINT python /comfy/main.py --listen 0.0.0.0 --port 8188