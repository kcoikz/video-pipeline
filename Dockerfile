FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg git wget curl && rm -rf /var/lib/apt/lists/*

# ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI /root/ComfyUI
RUN pip install -r /root/ComfyUI/requirements.txt

# Chatterbox TTS Server
RUN git clone https://github.com/devnen/Chatterbox-TTS-Server /root/Chatterbox-TTS-Server
RUN cd /root/Chatterbox-TTS-Server && pip install -r requirements.txt
RUN pip install s3tokenizer librosa pydub resemble-perth diffusers safetensors

# Gateway server
COPY runpod-server/ /root/runpod-server/
RUN pip install -r /root/runpod-server/requirements.txt

# ComfyUI model paths → Network Volume
COPY extra_model_paths.yaml /root/ComfyUI/extra_model_paths.yaml

RUN chmod +x /root/runpod-server/start.sh /root/runpod-server/install_models.sh

EXPOSE 5002 22
CMD ["bash", "/root/runpod-server/start.sh"]
