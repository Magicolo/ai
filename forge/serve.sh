#!/bin/bash

folder="$(realpath $(dirname $0))"
wait=0.5
timeout=10

docker volume create forge &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" build forge || exit $?
forge=$(docker compose --file "$folder/docker-compose.yml" run --service-ports --detach --rm forge) || exit $?

# Waiting for healthy container.
time=$(date +%s)
while [ $(($(date +%s) - time)) -le "$timeout" ]; do
    if [ $(docker inspect --format {{.State.Health.Status}} "$forge") == "healthy" ]; then
        break
    fi
    sleep "$wait"        
done

echo "$forge"