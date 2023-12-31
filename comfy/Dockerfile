FROM ubuntu

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl python3 python3-pip python-is-python3 ffmpeg
RUN git clone --recursive https://github.com/comfyanonymous/ComfyUI.git /comfy
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/ltdrdata/ComfyUI-Manager.git
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
RUN cd /comfy/custom_nodes && git clone --recursive https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install -r comfy/requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
HEALTHCHECK --interval=5s --timeout=30s --start-period=5s --retries=10 CMD curl http://localhost:8188
ENTRYPOINT python /comfy/main.py --listen 0.0.0.0 --port 8188