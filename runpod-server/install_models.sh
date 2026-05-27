#!/bin/bash
set -e

MODELS=/workspace/models
mkdir -p $MODELS/unet $MODELS/vae $MODELS/clip

echo "=== Downloading FLUX.1 Schnell fp8 (~12GB) ==="
[ ! -f "$MODELS/unet/flux1-schnell.safetensors" ] && \
    wget -q --show-progress -O "$MODELS/unet/flux1-schnell.safetensors" \
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8-e4m3fn.safetensors"

echo "=== Downloading VAE (~160MB) ==="
[ ! -f "$MODELS/vae/ae.safetensors" ] && \
    wget -q --show-progress -O "$MODELS/vae/ae.safetensors" \
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux-vae-bf16.safetensors"

echo "=== Downloading T5 text encoder (~4.9GB) ==="
[ ! -f "$MODELS/clip/t5xxl_fp8_e4m3fn.safetensors" ] && \
    wget -q --show-progress -O "$MODELS/clip/t5xxl_fp8_e4m3fn.safetensors" \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"

echo "=== Downloading CLIP L (~246MB) ==="
[ ! -f "$MODELS/clip/clip_l.safetensors" ] && \
    wget -q --show-progress -O "$MODELS/clip/clip_l.safetensors" \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"

echo "=== All models ready ==="
