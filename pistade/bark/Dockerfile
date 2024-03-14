FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git curl pkg-config ffmpeg portaudio19-dev
RUN pip install --upgrade pip scipy git+https://github.com/suno-ai/bark.git git+https://github.com/huggingface/transformers.git