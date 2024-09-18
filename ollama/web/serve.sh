#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create ollama || exit $?
docker volume create web || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm --detach --service-ports web