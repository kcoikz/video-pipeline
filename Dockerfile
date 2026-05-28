FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg git wget curl && rm -rf /var/lib/apt/lists/*

# ComfyUI (pin to known-good commit to avoid surprise breakage)
RUN git clone https://github.com/comfyanonymous/ComfyUI /root/ComfyUI && \
    cd /root/ComfyUI && git checkout v0.3.43
# Install ComfyUI deps but exclude torch/torchvision/torchaudio to preserve CUDA build
RUN grep -vE '^torch(vision|audio)?[>=<! ]|^torch$' /root/ComfyUI/requirements.txt > /tmp/comfy_nodeps.txt && \
    pip install -r /tmp/comfy_nodeps.txt

# Chatterbox TTS Server
RUN git clone https://github.com/devnen/Chatterbox-TTS-Server /root/Chatterbox-TTS-Server
# Install Chatterbox-TTS-Server deps (same torch exclusion)
RUN grep -vE '^torch(vision|audio)?[>=<! ]|^torch$' /root/Chatterbox-TTS-Server/requirements.txt > /tmp/ctts_nodeps.txt && \
    pip install -r /tmp/ctts_nodeps.txt
RUN pip install s3tokenizer librosa pydub resemble-perth diffusers safetensors

# Install the Chatterbox TTS core library (resemble-ai/chatterbox)
RUN pip install git+https://github.com/resemble-ai/chatterbox

# Reinstall CUDA-enabled PyTorch last so nothing overwrites it
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Gateway server
COPY runpod-server/ /root/runpod-server/
RUN pip install -r /root/runpod-server/requirements.txt

# ComfyUI model paths → Network Volume
COPY extra_model_paths.yaml /root/ComfyUI/extra_model_paths.yaml

RUN chmod +x /root/runpod-server/start.sh /root/runpod-server/install_models.sh

EXPOSE 5002 22
CMD ["bash", "/root/runpod-server/start.sh"]
