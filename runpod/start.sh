#!/bin/bash
set -e

echo "============================================"
echo "  Whisper LoRA Training Pod — Starting..."
echo "============================================"

# --- 1. S3 credentials ---
if [ -z "$HETZNER_S3_ACCESS_KEY" ] || [ -z "$HETZNER_S3_SECRET_KEY" ]; then
    echo "ERROR: HETZNER_S3_ACCESS_KEY and HETZNER_S3_SECRET_KEY must be set!"
    exit 1
fi

HETZNER_S3_ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
HETZNER_S3_BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"

echo "S3 Endpoint: $HETZNER_S3_ENDPOINT"
echo "S3 Bucket:   $HETZNER_S3_BUCKET"

# Записати credentials для s3fs
echo "${HETZNER_S3_ACCESS_KEY}:${HETZNER_S3_SECRET_KEY}" > /etc/passwd-s3fs
chmod 600 /etc/passwd-s3fs

# --- 2. Монтування S3 ---
S3_MOUNT="/workspace/s3"
S3_CACHE="/tmp/s3cache"
mkdir -p "$S3_MOUNT" "$S3_CACHE"

echo "Mounting S3 bucket to $S3_MOUNT..."
s3fs "$HETZNER_S3_BUCKET" "$S3_MOUNT" \
    -o passwd_file=/etc/passwd-s3fs \
    -o url="$HETZNER_S3_ENDPOINT" \
    -o use_path_request_style \
    -o allow_other \
    -o use_cache="$S3_CACHE" \
    -o ensure_diskfree=1024 \
    -o parallel_count=5 \
    -o multipart_size=64 \
    -o max_stat_cache_size=10000 \
    -o connect_timeout=10 \
    -o retries=3

# Перевірка монтування
if mountpoint -q "$S3_MOUNT"; then
    echo "S3 mounted successfully!"
else
    echo "ERROR: S3 mount failed!"
    exit 1
fi

# --- 3. Структура директорій ---
BASE_DIR="$S3_MOUNT/multilingual-stt-general"
REPO_DIR="$BASE_DIR/multilingual-stt"
CV_DIR="$BASE_DIR/CommonVoice"

mkdir -p "$BASE_DIR" "$CV_DIR"

# --- 4. Git repo ---
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR" && git pull || echo "WARNING: git pull failed, using existing version"
else
    echo "Cloning repo..."
    git clone https://github.com/BlackVarmir/multilingual-stt.git "$REPO_DIR"
fi

# --- 5. Symlinks ---
ln -sfn "$BASE_DIR" /workspace/multilingual-stt-general
ln -sfn "$REPO_DIR" /workspace/multilingual-stt

# --- 6. Статус ---
echo ""
echo "============================================"
echo "  Pod Ready!"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "S3 Mount: $S3_MOUNT"
echo "Repo: $REPO_DIR"
echo "CommonVoice: $CV_DIR"
echo "Files on S3: $(ls "$BASE_DIR" | wc -l) items"
echo ""
echo "Usage:"
echo "  cd /workspace/multilingual-stt"
echo "  python src/asr/train_whisper.py"
echo "============================================"

# --- 7. Keep alive ---
sleep infinity
