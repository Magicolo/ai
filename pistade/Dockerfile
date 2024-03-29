FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git curl pkg-config ffmpeg portaudio19-dev
RUN pip install --upgrade pip sounddevice scipy pydub noisereduce ollama setuptools-rust flash-attn
RUN pip install --upgrade git+https://github.com/resemble-ai/resemble-enhance.git
RUN pip install --upgrade git+https://github.com/m-bain/whisperx.git
RUN pip install --upgrade git+https://github.com/suno-ai/bark.git
RUN pip install --upgrade git+https://github.com/huggingface/transformers.git