FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl ffmpeg python3 python3-pip python-is-python3 google-perftools melt gmic
# RUN pip install --requirement https://raw.githubusercontent.com/declare-lab/tango/master/requirements.txt
# RUN pip install --requirement https://raw.githubusercontent.com/declare-lab/tango/master/tango2/requirements.txt
RUN pip install transformers==4.27.0 accelerate==0.18.0 datasets==2.1.0 diffusers==0.18.2 einops==0.6.1 h5py==3.8.0 huggingface_hub==0.13.3 importlib_metadata==6.3.0 librosa==0.9.2 matplotlib==3.5.2 numpy==1.23.0 omegaconf==2.3.0 packaging==23.1 pandas==1.4.1 progressbar33==2.4 protobuf==3.20.* resampy==0.4.2 scikit_image==0.19.3 scikit_learn==1.2.2 scipy==1.8.0 soundfile==0.12.1 ssr_eval==0.0.6 torchlibrosa==0.1.0 tqdm==4.63.1 wandb==0.12.14 ipython==8.12.0
RUN pip install jax==0.4.23 jaxlib==0.4.23
RUN pip install torch torchaudio torchvision
RUN pip install soundfile