#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create riffusion &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" build riffusion || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm --detach --service-ports riffusion || exit $?