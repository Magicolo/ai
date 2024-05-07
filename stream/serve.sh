#!/bin/bash

demo=$1
folder="$(realpath $(dirname $0))"
service=$([ -z "$demo" ] && echo "stream" || [[ $demo == stream* ]] && echo "$demo" || echo "stream-$demo")

docker volume create stream || exit $?
docker compose --file "$folder/docker-compose.yml" build "$service" || exit $?
docker compose --file "$folder/docker-compose.yml" run --rm "$service" || exit $?