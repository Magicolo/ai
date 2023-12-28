#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create flowise &> /dev/null || exit $?
docker compose --file "$folder/docker-compose.yml" --file "$folder/../ollama/docker-compose.yml" up --force-recreate --remove-orphans