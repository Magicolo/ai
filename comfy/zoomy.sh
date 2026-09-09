#!/bin/bash

folder="$(realpath $(dirname $0))"
docker compose --file "$folder/docker-compose.yml" run --build --rm --detach --service-ports --no-deps zoomy
