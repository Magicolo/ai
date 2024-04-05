FROM python

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git curl pkg-config ffmpeg portaudio19-dev
RUN pip install --upgrade pip sounddevice scipy pydub noisereduce ollama
CMD sleep infinity