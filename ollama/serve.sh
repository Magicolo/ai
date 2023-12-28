#!/bin/bash

port="${1:-11434}"

docker build --tag ollama "$folder" &> /dev/null
docker volume create ollama &> /dev/null
if [ "$port" -le 0 ]; then
    docker run  --gpus all --rm --detach --volume "ollama:/root/.ollama" ollama
else
    docker run  --gpus all --rm --detach --publish "$port:11434" --volume "ollama:/root/.ollama" ollama
fi