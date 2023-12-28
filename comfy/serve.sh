#!/bin/bash

port="${1:-8188}"
folder="$(realpath $(dirname $0))"
output="$folder/output"
wait=0.5
timeout=50
entry=$(cat "$folder/entry.sh")

docker build --tag comfy "$folder" &> /dev/null || exit $?
docker volume create comfy &> /dev/null || exit $?
if [ "$port" -le 0 ]; then
    comfy=$(docker run --interactive --gpus all --rm --detach \
        --volume "$output:/comfy/output" \
        --volume comfy:/comfy/models \
        comfy bash -c "$entry")
else
    comfy=$(docker run --interactive --gpus all --rm --detach \
        --publish "$port:8188" \
        --volume "$output:/comfy/output" \
        --volume comfy:/comfy/models \
        comfy bash -c "$entry")
fi

# Waiting for healthy container.
time=$(date +%s)
while [ $(($(date +%s) - time)) -le "$timeout" ]; do
    if [ $(docker inspect --format {{.State.Health.Status}} "$comfy") == "healthy" ]; then
        break
    fi
    sleep "$wait"        
done

echo "$comfy"