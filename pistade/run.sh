#!/bin/bash

folder="$(realpath $(dirname $0))"
docker volume create pistade
docker volume create enhance
docker compose --file "$folder/docker-compose.yml" --file "$folder/../ollama/docker-compose.yml" up --remove-orphans --build