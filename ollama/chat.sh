#!/bin/bash

model="$1"
[ -z "$model" ] && { echo "Missing model name." >&2; exit 1; }

folder="$(realpath $(dirname $0))"
ollama=$(bash "$folder/serve.sh" 0) || exit $?
stop() { docker stop "$ollama" &> /dev/null; }
trap stop EXIT
docker exec --interactive --tty "$ollama" ollama run "$model" || exit $?