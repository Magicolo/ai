FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl ffmpeg python3 python3-pip python-is-python3 google-perftools melt gmic
RUN pip install torch torchaudio torchtext torchvision av einops flashy hydra-core hydra_colorlog julius num2words numpy sentencepiece spacy huggingface_hub tqdm transformers xformers demucs librosa gradio torchmetrics encodec protobuf
RUN pip install pydub