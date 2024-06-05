#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create tango &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" build tango || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm --detach --service-ports tango || exit $?