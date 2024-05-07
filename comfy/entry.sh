#!/bin/bash

folder="$(realpath $(dirname $0))"
models="$folder/models"
checkpoints="$models/checkpoints"
loras="$models/loras"
vaes="$models/vae"
nets="$models/controlnet"

download() {
    if [ ! -e "$1" ]; then 
        curl --location --create-dirs --output "$1" "$2" || rm --force "$1"
    fi
}

download "$checkpoints/TURBO_SDXL_Original.safetensors" https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors?download=true &
download "$checkpoints/TURBO_TurboVision.safetensors" https://civitai.com/api/download/models/273102 &
download "$checkpoints/TURBO_UnstableDiffusers.safetensors" https://civitai.com/api/download/models/247214 &
download "$checkpoints/TURBO_DreamShaper.safetensors" https://civitai.com/api/download/models/351306 &
download "$checkpoints/SDXL_Juggernaut.safetensors" https://civitai.com/api/download/models/456194 &
download "$checkpoints/SDXL_UnstableDiffusers.safetensors" https://civitai.com/api/download/models/395107?type=Model&format=SafeTensor&size=pruned&fp=fp16 &
download "$checkpoints/SDXL_Colossus.safetensors" https://civitai.com/api/download/models/459216 &
download "$checkpoints/SDXL_ThinkDiffusion.safetensors" https://civitai.com/api/download/models/190908?type=Model&format=SafeTensor&size=full&fp=fp16 &
download "$checkpoints/LIGHTNING_RealVis.safetensors" https://civitai.com/api/download/models/361593 &
download "$vaes/SD15_ClearVae.safetensors" https://civitai.com/api/download/models/88156?type=Model&format=SafeTensor &
download "$vaes/SD15_Kl-F8-Anime.safetensors" https://civitai.com/api/download/models/28569?type=Model&format=Other &
download "$vaes/SD15_Vae-Ft-Mse840000.safetensors" https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true &
download "$vaes/SDXL_Vae.safetensors" https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors?download=true &
download "$nets/SD15_Control_v11f1p_Depth.pth" https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth?download=true &
download "$nets/SD15_Control_v11f1p_Depth.yaml" https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.yaml?download=true &
download "$nets/SD15_Control_v11p_Lineart.pth" https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth?download=true &
download "$nets/SD15_Control_v11p_Lineart.yaml" https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.yaml?download=true &
download "$nets/SD15_Control_v11p_Openpose.pth" https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth?download=true &
download "$nets/SD15_Control_v11p_Openpose.yaml" https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.yaml?download=true &
download "$nets/SDXL_Diffusers_Canny.safetensors" https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors?download=true &
download "$nets/SDXL_Diffusers_Depth.safetensors" https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_full.safetensors?download=true &
download "$nets/SDXL_T2i-Adapter_Diffusers_Lineart.safetensors" https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_lineart.safetensors?download=true &
download "$nets/SDXL_T2i-Adapter_Diffusers_Openpose.safetensors" https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_openpose.safetensors?download=true &
download "$nets/SDXL_T2i-Adapter_Diffusers_Sketch.safetensors" https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_sketch.safetensors?download=true &
download "$loras/SDXL_Lora.safetensors" https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true &
download "$loras/SD15_Lora.safetensors" https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true &
python "$folder/main.py" --listen 0.0.0.0 --port 8188