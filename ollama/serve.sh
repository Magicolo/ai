#!/bin/bash

port="${1:-11434}"
volume="ollama:/root/.ollama"
image="ollama/ollama"
if [ "$port" -le 0 ]; then
    docker run  --gpus all --rm --detach --volume "$volume" "$image"
else
    docker run  --gpus all --rm --detach --publish "$port:11434" --volume "$volume" "$image"
fi