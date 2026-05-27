#!/bin/bash
set -e

export HF_HOME=/workspace/.hf_cache
mkdir -p /workspace/.hf_cache /workspace/models/unet /workspace/models/vae /workspace/models/clip

echo "=== Checking FLUX models ==="
if [ ! -f "/workspace/models/unet/flux1-schnell.safetensors" ]; then
    echo "=== Models not found, downloading... ==="
    bash /root/runpod-server/install_models.sh
fi

echo "=== Starting Chatterbox TTS Server (port 8004) ==="
cd /root/Chatterbox-TTS-Server
python server.py &

echo "=== Starting ComfyUI (port 8188) ==="
cd /root/ComfyUI
python main.py --listen 127.0.0.1 --port 8188 &

echo "=== Waiting 30s for services to initialize ==="
sleep 30

echo "=== Starting gateway (port 5002) ==="
cd /root/runpod-server
uvicorn server:app --host 0.0.0.0 --port 5002
