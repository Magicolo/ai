#!/bin/bash

folder="$(realpath $(dirname $0))"
docker build --tag coqui "$folder" &> /dev/null || exit $?
docker volume create coqui &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm coqui