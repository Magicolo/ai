#!/bin/bash

model="$1"
[ -z "$model" ] && { echo "Missing model name." >&2; exit 1; }
prompt="$2"
[ -z "$prompt" ] && { echo "Missing prompt." >&2; exit 1; }
image="$3"
[ -z "$image" ] && { echo "Missing image path." >&2; exit 1; }

folder="$(realpath $(dirname $0))"
ollama=$(bash "$folder/serve.sh" 0) || exit $?
stop() { docker stop "$ollama" &> /dev/null; }
trap stop EXIT

images=$(base64 -w 0 "$image" | sed 's/.*/["&"]/')
data='{
    "model": '"$(jq -R <<< "$model")"',
    "prompt": '"$(jq -Rs <<< "$prompt")"',
    "stream": false,
    "images": '"$(jq <<< "$images")"'
}'
response=$(echo "$data" | docker exec --interactive "$ollama" curl localhost:11434/api/generate --silent --data @-)
echo "$response" | jq  -r ".response" || exit $?