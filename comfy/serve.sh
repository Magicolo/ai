#!/bin/bash

port="${1:-8188}"
folder="$(realpath $(dirname $0))"
output="$folder/output"
wait=0.5
timeout=10

docker volume create comfy &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" build comfy || exit $?
if [ "$port" -le 0 ]; then
    comfy=$(docker compose --file "$folder/docker-compose.yml" run --rm --detach comfy) || exit $?
else
    comfy=$(docker compose --file "$folder/docker-compose.yml" run --rm --detach --service-ports comfy) || exit $?
fi

# Waiting for healthy container.
time=$(date +%s)
while [ $(($(date +%s) - time)) -le "$timeout" ]; do
    if [ $(docker inspect --format {{.State.Health.Status}} "$comfy") == "healthy" ]; then
        break
    fi
    sleep "$wait"        
done

echo "$comfy"