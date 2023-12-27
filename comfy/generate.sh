#!/bin/bash

graph="$1"
[ -z "$graph" ] && { echo "Missing comfy graph." >&2; exit 1; }
pattern="${2:-.}"
wait=0.5
timeout=50
folder="$(realpath $(dirname $0))"
output="$folder/output"

comfy=$(bash "$folder/serve.sh" 0) || exit $?
stop() { docker stop "$comfy" &> /dev/null; }
trap stop EXIT

old=$(ls -t "$output" | grep "$pattern" | head -n 1)
response=$(docker exec "$comfy" curl localhost:8188/prompt --silent --data "$graph") || exit $?

# Waiting for image generation.
time=$(date +%s)
while [ $(($(date +%s) - time)) -le "$timeout" ]; do
    new=$(ls -t "$output" | grep "$pattern" | head -n 1)
    if [ "$old" != "$new" ]; then
        echo "$output/$new"
        exit 0
    fi
    sleep "$wait"
done
exit 1