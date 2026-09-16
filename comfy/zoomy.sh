#!/bin/bash

folder="$(realpath $(dirname $0))"
# Standalone zoomy: the image carries the whole engine (frames, interp, ACE,
# MMAudio), models live in the zoomy_models volume, outputs land in
# ./Zoomy/output. No comfy involved. First launch ever needs one provisioning
# run: CIVITAI_API_KEY=$(cat civit-ai-api-key) docker compose --file
# "$folder/docker-compose.yml" run --rm -v "$folder/Comfy/input:/seed-source:ro"
# zoomy python scripts/download_models.py --seed-source /seed-source
docker compose --file "$folder/docker-compose.yml" run --build --rm --detach --service-ports zoomy
