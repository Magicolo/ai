#!/bin/bash

folder="$(realpath $(dirname $0))"
docker build --tag nemo "$folder" &> /dev/null || exit $?
docker volume create nemo &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm nemo