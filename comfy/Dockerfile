FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl ffmpeg python3 python3-pip python-is-python3 google-perftools melt gmic
RUN pip install --upgrade pip opencv-python gradio blosc posix_ipc
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --requirement https://github.com/comfyanonymous/ComfyUI/raw/master/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Pack/Main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/ltdrdata/ComfyUI-Inspire-Pack/main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/Fannovel16/comfyui_controlnet_aux/main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/Fannovel16/ComfyUI-Frame-Interpolation/main/requirements-with-cupy.txt
RUN pip install --requirement https://raw.githubusercontent.com/Kosinkadink/ComfyUI-VideoHelperSuite/main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/rgthree/rgthree-comfy/main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/cubiq/ComfyUI_essentials/main/requirements.txt
RUN pip install --requirement https://raw.githubusercontent.com/kijai/ComfyUI-SUPIR/main/requirements.txt
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip install --upgrade spandrel xformers tensorrt ultralytics flash-attn transformers
RUN apt install --yes v4l-utils
RUN conda update --yes ffmpeg