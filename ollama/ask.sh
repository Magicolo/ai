#!/bin/bash

model="$1"
[ -z "$model" ] && { echo "Missing model name." >&2; exit 1; }
prompt="$2"
[ -z "$prompt" ] && { echo "Missing prompt." >&2; exit 1; }

folder="$(realpath $(dirname $0))"
ollama=$(bash "$folder/serve.sh" 0) || exit $?
stop() { docker stop "$ollama" &> /dev/null; }
trap stop EXIT
docker exec "$ollama" ollama run "$model" "$prompt" || exit $?
