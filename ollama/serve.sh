#!/bin/bash

port="${1:-11434}"
folder="$(realpath $(dirname $0))"

docker volume create ollama || exit $?
docker compose --file "$folder/docker-compose.yml" build ollama || exit $?
if [ "$port" -le 0 ]; then
    docker compose --file "$folder/docker-compose.yml" run --rm --detach ollama
else
    docker compose --file "$folder/docker-compose.yml" run --rm --detach --publish "$port:11434" ollama
fi