FROM nvcr.io/nvidia/nemo:24.01.framework

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git curl pkg-config ffmpeg portaudio19-dev
RUN git clone https://github.com/NVIDIA/NeMo.git /NeMo
RUN pip install --upgrade pip sounddevice wave lhotse