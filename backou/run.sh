#!/bin/bash

positive="ultra detailed, surreal, abstract, conceptual, masterpiece"
negative="blurry, smooth, simple, plain, cute, naked, nude, nudity, sexy"
width=1152
height=896
steps=10
guidance=2.5
model="TURBO_DreamShaper.safetensors"
sampler="dpmpp_sde_gpu"
scheduler="karras"
upscaler="RealESRGAN_x2.pth"
upscale=2
skip=""
prompt=""
memory=25
keep=false
prompter="solar:10.7b-instruct-v1-q8_0"
descriptor="llava:13b"

while [ $# -gt 0 ]; do
    key="${1#--}"
    value="$2"
    declare -p "$key" &>/dev/null || { echo "Unknown option '$key'." >&2; exit 1; }
    [ "$value" ] || { echo "Missing value for '$value'." >&2; exit 1; }
    declare -g "$key=$value"
    shift 2
done

folder="$(realpath $(dirname $0))"
prefix="background"
history="$folder/history"

if [ -z "$prompt" ]; then
    if [ "$skip" == "prompt" ]; then
        echo "=> Skipping prompt generation."
        [ -z "$prompt" ] && prompt=$(shuf "$history" -n 1)
    else
        touch "$history"
        pre=$(("$memory" * 2))
        words=$(grep -E -o '\b[a-zA-Z]+\b' "$history" | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -n | awk '{print $2}') || exit $?
        inspire=$(echo -e "$words" | awk 'length($0) > 3' | head -n "$pre" | shuf -n "$memory" | sed ':a;N;$!ba;s/\n/, /g') || exit $?
        forbid=$(echo -e "$words" | awk 'length($0) > 5' | tail -n "$pre" | shuf -n "$memory" | sed ':a;N;$!ba;s/\n/, /g') || exit $?
        instructions="
You are a wildly creative and highly skilled prompt engineer for a txt2img AI.
You are strictly only allowed to write a single prompt; no introduction, no comment, no analysis; only the one prompt itself.
The prompt may relate to these inspiration words: $inspire.
The prompt must NOT relate to any of these forbidden, banned, disallowed words: $forbid.
The prompt must consist of specific, unambiguous, clear, obvious and detailed visual indications.
The prompt must be weird, menacing, creepy, intricate, lively, impossible, concrete, exalted, surreal, experimental.
The prompt must describe the colors, shapes, characters, themes, creatures, environments, places that will be in the image with lots of adjectives.
The prompt must begin with a parenthesized short description of the distinct, exotic and niche artistic visual style of the image.
Write a single succinct image prompt that strictly comply with the above rules.

Prompt:
"
        echo "=> Generating prompt..."
        echo "==> Forbid: $forbid"
        echo "==> Inspire: $inspire"
        prompt=$(bash "$folder/../ollama/ask.sh" "$prompter" "$instructions" | grep "[a-zA-Z]" | tr -d '"' | tr -d "'" | tr '\n' ' ' | tr '\r' ' ' | tr '\t' ' ') || exit $?
        line=$(echo -e "$prompt" | tr '\n' ' ')
        echo -e "$line" >> "$history"
    fi
fi

echo "=> Prompt: $prompt"

if [ "$skip" == "image" ]; then 
    echo "=> Skipping image generation."
else
    echo "=> Generating image..."
    graph=$(export prompt="$prompt" forbid="$forbid" seed=$RANDOM noise_seed=$RANDOM positive="$positive" negative="$negative" width="$width" height="$height" steps="$steps" guidance="$guidance" model="$model" sampler="$sampler" scheduler="$scheduler" upscaler="$upscaler" upscale="$upscale" prefix="$prefix" && cat "$folder/template.json" | envsubst | tr '\n' ' ') || exit $?
    image=$(bash "$folder/../comfy/generate.sh" "$graph" "^$prefix") || exit $?
    echo "==> Image: $image"
    background="$folder/background.png"
    now=$(date +"%Y%m%d%H%M%S")
    [ "$keep" == "true" ] && cp -f "$image" "$folder/keep/background-$now.png"
    mv -f "$image" "$background" || exit $?
    gsettings set org.gnome.desktop.background picture-uri "file://$background" || exit $?
    
    if [ "$skip" == "describe" ]; then
        echo "=> Skipping image description."
    else 
        echo "=> Generating description..."
        describe="Write an exhaustive list of lowercase keywords that describe the subjects, characters, colors, shapes, creatures, places, artistic styles, decors, themes, environments and ambiances of the image."
        description=$(bash "$folder/../ollama/describe.sh" "$descriptor" "$describe" "$background" | grep "[a-zA-Z]" | tr -d '"' | tr '\n' ' ') || exit $?
        echo "==> Description: $description"
        line=$(echo -e "$description" | tr '\n' ' ')
        echo -e "$line" >> "$history"
    fi
fi

echo "=> Done."
echo ""
