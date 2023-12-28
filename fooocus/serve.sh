#!/bin/bash

folder="$(realpath $(dirname $0))"
output="$folder/output"

docker build --tag fooocus "$folder" &> /dev/null || exit $?
docker volume create fooocus &> /dev/null || exit $?
docker run --interactive --gpus all --rm --detach \
    --publish 7860:7860 \
    --volume "$output:/fooocus/outputs" \
    --volume fooocus:/fooocus/models fooocus