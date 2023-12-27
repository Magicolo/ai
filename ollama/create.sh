#!/bin/bash

environment=$(declare -p)
source="$1"
[ -z "$source" ] && { echo "Missing source repository." >&2; exit 1; }
name="$(basename $source)"
quant="q8_0"

shift
while [ $# -gt 0 ]; do
    key="${1#--}"
    value="$2"
    declare -p "$key" &>/dev/null || { echo "Unknown option '$key'." >&2; exit 1; }
    [ "$value" ] || { echo "Missing value for '$value'." >&2; exit 1; }
    declare -g "$key=$value"
    shift 2
done

folder="$(realpath $(dirname $0))"
target="$folder/$name"
git lfs install
if [ -e $target/.git ]; then
    cd "$target"
    git pull || exit $?
else 
    git clone "$source" "$target" --recursive || exit $?
    cd "$target"
fi

bash "$folder/quantize.sh" "$target" "$quant" || exit $?

[ ! -f "Modelfile" ] && echo "FROM ./$quant.bin" > "Modelfile"
ollama=$(docker run --rm --detach --volume "$target:/model" --volume ollama:/root/.ollama --workdir /model ollama/ollama) || exit $?
stop() { docker stop "$ollama" &> /dev/null; }
trap stop EXIT
docker exec "$ollama" ollama create "$name" || exit $?