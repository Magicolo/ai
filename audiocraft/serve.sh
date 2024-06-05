#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create audiocraft &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" build audiocraft || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm --detach --service-ports audiocraft || exit $?