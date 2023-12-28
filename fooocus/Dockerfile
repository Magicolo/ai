FROM ubuntu

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git git-lfs curl python3 python3-pip python-is-python3 ffmpeg
RUN git clone --recursive https://github.com/lllyasviel/Fooocus.git /fooocus
RUN pip install --upgrade pip
RUN pip install -r fooocus/requirements_versions.txt
# HEALTHCHECK --interval=1s --timeout=30s --start-period=1s --retries=10 CMD curl localhost:7860
ENTRYPOINT python /fooocus/entry_with_update.py --preset realistic