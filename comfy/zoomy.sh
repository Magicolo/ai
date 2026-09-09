#!/bin/bash

folder="$(realpath $(dirname $0))"
# The zoomy service depends on comfy (see depends_on in docker-compose.yml).
# `run --no-deps` below deliberately skips compose-managed startup so there is
# exactly one way to run comfy: serve.sh (fixed --name comfy, which is also
# the DNS name zoomy uses). Spawn it here when it is not running.
if [ "$(docker inspect --format '{{.State.Running}}' comfy 2>/dev/null)" != "true" ]; then
  docker rm --force comfy 2>/dev/null || true
  "$folder/serve.sh" || exit $?
fi
docker compose --file "$folder/docker-compose.yml" run --build --rm --detach --service-ports --no-deps zoomy
