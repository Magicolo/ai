#!/bin/bash

folder="$(realpath $(dirname $0))"

docker volume create fooocus &> /dev/null || exit $?
docker compose --file "$folder/Fooocus/docker-compose.yml" build fooocus || exit $
# docker compose --file "$folder/docker-compose.yml" build --quiet fooocus &> /dev/null || exit $
docker compose --file "$folder/Fooocus/docker-compose.yml" run --detach fooocus || exit $