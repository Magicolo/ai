FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl wget make tar build-essential unzip pkg-config
RUN apt install --yes libgl1-mesa-dev libglib2.0-0 libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt install --yes python3-dev python3-pip python-is-python3
RUN apt install --yes ffmpeg portaudio19-dev melt gmic
RUN apt clean

RUN pip install --upgrade torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install --upgrade packaging
RUN pip install --upgrade streamdiffusion[tensorrt] setuptools-rust packaging flash-attn transformers
RUN pip install --upgrade pip numpy sounddevice scipy protobuf sentencepiece scikit-learn Cython matplotlib tqdm opencv-python gradio
RUN git clone --recursive https://github.com/cumulo-autumn/StreamDiffusion.git /stream
RUN pip install --requirement /stream/demo/realtime-txt2img/requirements.txt
RUN pip install --requirement /stream/demo/realtime-img2img/requirements.txt
RUN pip install --requirement /stream/demo/vid2vid/requirements.txt
RUN pip install --upgrade polygraphy onnx_graphsurgeon
# RUN cd /stream && python setup.py develop easy_install streamdiffusion[tensorrt]
# RUN cd /stream && python -m streamdiffusion.tools.install-tensorrt