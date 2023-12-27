target="$1"
[ -z "$target" ] && { echo "Missing target repository." >&2; exit 1; }
quant="${2:-q8_0}"

ollama=$(docker run --rm --volume "$target:/model" ollama/quantize -q "$quant" /model) || exit $?
stop() { docker stop "$ollama" &> /dev/null; }
trap stop EXIT
