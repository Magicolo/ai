FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl ffmpeg python3 python3-pip python-is-python3 google-perftools melt gmic
RUN pip install --upgrade pip opencv-python