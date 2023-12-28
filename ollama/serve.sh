#!/bin/bash

port="${1:-11434}"
folder="$(realpath $(dirname $0))"

docker build --tag ollama "$folder" &> /dev/null || exit $?
docker volume create ollama &> /dev/null || exit $?
if [ "$port" -le 0 ]; then
    docker run  --gpus all --rm --detach --volume "ollama:/root/.ollama" ollama
else
    docker run  --gpus all --rm --detach --publish "$port:11434" --publish 8000:8000 --volume "ollama:/root/.ollama" ollama
fi