#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create canary
docker build --tag canary "$folder/canary"
docker volume create bark
docker build --tag bark "$folder/bark"
docker compose --file "$folder/docker-compose.yml" up