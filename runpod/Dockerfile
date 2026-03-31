FROM runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2404

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
