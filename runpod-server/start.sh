#!/bin/bash

export HF_HOME=/workspace/.hf_cache
mkdir -p /workspace/.hf_cache /workspace/models/unet /workspace/models/vae /workspace/models/clip /workspace/output /workspace/logs

# ── Pull latest code into /workspace/code ─────────────────────────
CODE_DIR=/workspace/code
if [ ! -d "$CODE_DIR/.git" ]; then
    echo "=== Cloning repo to $CODE_DIR ==="
    rm -rf "$CODE_DIR"
    git clone https://github.com/kcoikz/video-pipeline.git "$CODE_DIR"
fi
cd "$CODE_DIR" && git pull origin main 2>&1 | tail -3

# ── Check FLUX models ─────────────────────────────────────────────
echo "=== Checking FLUX models ==="
if [ ! -f "/workspace/models/unet/flux1-schnell.safetensors" ]; then
    echo "=== Models not found, downloading... ==="
    bash "$CODE_DIR/runpod-server/install_models.sh"
fi

# ── Start Chatterbox (port 8004) ──────────────────────────────────
echo "=== Starting Chatterbox TTS Server (port 8004) ==="
cd /root/Chatterbox-TTS-Server
python server.py > /workspace/logs/chatterbox.log 2>&1 &
CHATTERBOX_PID=$!
echo "Chatterbox PID=$CHATTERBOX_PID"

# ── Start ComfyUI (port 8188) ─────────────────────────────────────
echo "=== Starting ComfyUI (port 8188) ==="
cd /root/ComfyUI
python main.py --listen 127.0.0.1 --port 8188 > /workspace/logs/comfyui.log 2>&1 &
COMFYUI_PID=$!
echo "ComfyUI PID=$COMFYUI_PID"

# ── Wait for Chatterbox to become ready (up to 15 min) ────────────
echo "=== Waiting for Chatterbox to become ready (up to 15 min) ==="
READY=0
for i in $(seq 1 180); do
    if ! kill -0 $CHATTERBOX_PID 2>/dev/null; then
        echo "ERROR: Chatterbox process $CHATTERBOX_PID died at attempt $i"
        echo "--- last 40 lines of chatterbox.log ---"
        tail -40 /workspace/logs/chatterbox.log
        break
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://127.0.0.1:8004/health" 2>/dev/null || echo "0")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Chatterbox ready after ${i}x5s (${i}*5=$((i*5))s)"
        READY=1
        break
    fi
    echo "Chatterbox not ready yet (attempt $i/180, http=$HTTP_CODE)"
    sleep 5
done

if [ "$READY" = "0" ]; then
    echo "WARNING: Chatterbox did not become ready — gateway will start anyway, TTS calls will fail until Chatterbox recovers"
fi

# ── Run uvicorn in loop, pulling fresh code each restart ──────────
echo "=== Starting gateway (port 5002) with auto-update loop ==="
while true; do
    cd "$CODE_DIR"
    echo "=== $(date): git pull ==="
    git pull origin main 2>&1 | tail -3
    cd "$CODE_DIR/runpod-server"
    echo "=== $(date): starting uvicorn ==="
    uvicorn server:app --host 0.0.0.0 --port 5002
    echo "=== $(date): uvicorn exited, restarting in 2s ==="
    sleep 2
done
