#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create comfy || exit $?
docker compose --file "$folder/docker-compose.yml" run --build --rm --detach --service-ports comfy