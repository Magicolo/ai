#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create stream || exit $?
docker compose --file "$folder/docker-compose.yml" build stream || exit $?
docker compose --file "$folder/docker-compose.yml" run --service-ports --rm stream || exit $?