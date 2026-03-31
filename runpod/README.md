# RunPod Template: Whisper LoRA Training with Hetzner S3

Custom Docker template for RunPod that uses Hetzner Object Storage (S3-compatible) instead of Network Volume.
All data persists on S3 — Pod can be terminated without data loss.

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
| Container Disk | `20 GB` |
| Volume Disk | `0 GB` (not needed, using S3) |
| Volume Mount Path | `/workspace` |

### 3. Set Environment Variables

In RunPod Template or Pod settings, add:

| Variable | Value |
|----------|-------|
| `HETZNER_S3_ACCESS_KEY` | Your Hetzner S3 access key |
| `HETZNER_S3_SECRET_KEY` | Your Hetzner S3 secret key |
| `HETZNER_S3_ENDPOINT` | `https://fsn1.your-objectstorage.com` |
| `HETZNER_S3_BUCKET` | `multilingual-stt` |

### 4. Create Pod

- Select the template
- Choose GPU: A100 80GB (recommended)
- Start Pod

### 5. SSH & Train

```bash
# Connect via SSH
ssh root@<pod-ip>

# Start training in tmux
tmux new -s train
cd /workspace/multilingual-stt
python src/asr/train_whisper.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t train
```

## S3 Structure

```
S3 bucket/
└── multilingual-stt-general/
    ├── multilingual-stt/              ← GitHub repo (auto-cloned)
    │   └── models/whisper-uk-lora/    ← Training checkpoints
    └── CommonVoice/                   ← Dataset archive
```

## Upload CommonVoice to S3

Before first training, upload the dataset archive:

```bash
# On any machine with awscli
aws s3 cp cv-corpus-24.0-2025-12-05-uk.tar.gz \
    s3://multilingual-stt/multilingual-stt-general/CommonVoice/ \
    --endpoint-url https://fsn1.your-objectstorage.com

# Or on the Pod (after S3 is mounted)
cp /path/to/cv-corpus-24.0-2025-12-05-uk.tar.gz /workspace/s3/multilingual-stt-general/CommonVoice/
```

## Paths on Pod

| Path | Description |
|------|-------------|
| `/workspace/s3/` | S3 bucket mount point |
| `/workspace/multilingual-stt/` | Symlink to repo on S3 |
| `/workspace/multilingual-stt-general/` | Symlink to base dir on S3 |

## Notes

- **No Network Volume needed** — S3 is the persistent storage
- **Pod is disposable** — terminate anytime, data stays on S3
- **s3fs caching** — local cache at `/tmp/s3cache` speeds up repeated reads
- **CommonVoice** — archive stored on S3, extract when needed for training
