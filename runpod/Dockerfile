FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# s3fs для монтування Hetzner Object Storage
RUN apt-get update && apt-get install -y --no-install-recommends \
    s3fs \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python залежності для Whisper LoRA тренування
RUN pip install --no-cache-dir \
    transformers \
    peft \
    "datasets<4.0" \
    accelerate \
    evaluate \
    jiwer \
    librosa \
    soundfile \
    awscli \
    boto3 \
    tensorboard \
    wandb \
    huggingface-hub \
    safetensors \
    torchaudio

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
