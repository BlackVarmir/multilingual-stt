# RunPod Template: Whisper LoRA Training with Hetzner S3

Custom Docker template for RunPod that uses Hetzner Object Storage (S3-compatible) for persistent data.
Pod can be terminated without data loss — checkpoints and datasets sync to/from S3.

## Quick Start

### 1. Build & Push Docker Image

```bash
cd runpod
docker build -t blackvarmir/whisper-lora-runpod:latest .
docker push blackvarmir/whisper-lora-runpod:latest
```

### 2. Create RunPod Template

Go to RunPod Dashboard → Templates → New Template:

| Field | Value |
|-------|-------|
| Template Name | `whisper-lora-s3` |
| Container Image | `blackvarmir/whisper-lora-runpod:latest` |
| Container Disk | `50 GB` |
| Volume Disk | `0 GB` (not needed, using S3) |

### 3. Set Environment Variables

In RunPod Template or Pod settings, add:

| Variable | Value |
|----------|-------|
| `HETZNER_S3_ACCESS_KEY` | Your Hetzner S3 access key |
| `HETZNER_S3_SECRET_KEY` | Your Hetzner S3 secret key |
| `HETZNER_S3_ENDPOINT` | `https://fsn1.your-objectstorage.com` |
| `HETZNER_S3_BUCKET` | Your bucket name |

### 4. Create Pod & Train

```bash
ssh root@<pod-ip>

tmux new -s train
cd /workspace/multilingual-stt
python src/asr/train_whisper.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t train
```

### 5. Save Results to S3

```bash
s3-upload        # upload checkpoints to S3
s3-upload-cv     # upload CommonVoice archive to S3
```

## How It Works

- **On startup**: clones/pulls repo, syncs CommonVoice and checkpoints from S3
- **During training**: everything runs locally on fast NVMe disk
- **After training**: run `s3-upload` to push checkpoints back to S3

## S3 Structure

```
S3 bucket/
└── multilingual-stt-general/
    ├── CommonVoice/                   ← Dataset archive
    └── checkpoints/                   ← Training checkpoints
```

## Helper Commands

| Command | Description |
|---------|-------------|
| `s3-upload` | Upload checkpoints to S3 |
| `s3-upload-cv` | Upload CommonVoice archive to S3 |
