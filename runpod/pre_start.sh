#!/bin/bash
echo "============================================"
echo "  Storage Box Setup"
echo "============================================"

# --- Встановити залежності ---
apt-get update -qq && apt-get install -y -qq cifs-utils s3fs ffmpeg > /dev/null 2>&1
pip install -q peft "datasets<4.0" accelerate evaluate jiwer librosa soundfile awscli torchaudio 2>/dev/null

MOUNT_POINT="/workspace/storage"
MOUNTED=false

# =============================================
# Спосіб 1: CIFS mount (Storage Box)
# =============================================
if [ -n "$STORAGE_USER" ] && [ -n "$STORAGE_PASS" ]; then
    STORAGE_HOST="${STORAGE_HOST:-${STORAGE_USER}.your-storagebox.de}"
    echo "[1/3] Trying CIFS mount..."
    mkdir -p "$MOUNT_POINT"
    mount -t cifs "//${STORAGE_HOST}/backup" "$MOUNT_POINT" \
        -o "user=${STORAGE_USER},pass=${STORAGE_PASS},uid=0,gid=0,file_mode=0777,dir_mode=0777" 2>&1 \
        && MOUNTED=true && echo "CIFS mount OK!" \
        || echo "CIFS mount failed."
fi

# =============================================
# Спосіб 2: s3fs FUSE mount
# =============================================
if [ -n "$HETZNER_S3_ACCESS_KEY" ] && [ -n "$HETZNER_S3_SECRET_KEY" ] && [ "$MOUNTED" = false ]; then
    HETZNER_S3_ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
    HETZNER_S3_BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"
    echo "[2/3] Trying s3fs mount..."

    [ -e /dev/fuse ] || mknod /dev/fuse c 10 229 2>/dev/null || true
    chmod 666 /dev/fuse 2>/dev/null || true

    echo "${HETZNER_S3_ACCESS_KEY}:${HETZNER_S3_SECRET_KEY}" > /etc/passwd-s3fs
    chmod 600 /etc/passwd-s3fs
    mkdir -p "$MOUNT_POINT" /tmp/s3cache

    s3fs "${HETZNER_S3_BUCKET}" "$MOUNT_POINT" \
        -o passwd_file=/etc/passwd-s3fs \
        -o url="$HETZNER_S3_ENDPOINT" \
        -o use_path_request_style \
        -o use_cache=/tmp/s3cache \
        -o parallel_count=5 \
        -o allow_other 2>&1 \
        && MOUNTED=true && echo "s3fs mount OK!" \
        || echo "s3fs mount failed."
fi

# =============================================
# Спосіб 3: awscli sync (fallback)
# =============================================
if [ -n "$HETZNER_S3_ACCESS_KEY" ] && [ -n "$HETZNER_S3_SECRET_KEY" ] && [ "$MOUNTED" = false ]; then
    HETZNER_S3_ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
    HETZNER_S3_BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"
    echo "[3/3] Using awscli sync..."

    mkdir -p ~/.aws
    cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = ${HETZNER_S3_ACCESS_KEY}
aws_secret_access_key = ${HETZNER_S3_SECRET_KEY}
EOF
    export AWS_ENDPOINT_URL="$HETZNER_S3_ENDPOINT"

    mkdir -p /workspace/multilingual-stt/data/common_voice
    mkdir -p /workspace/multilingual-stt/models/whisper-uk-lora

    aws s3 sync "s3://${HETZNER_S3_BUCKET}/multilingual-stt-general/CommonVoice/" \
        /workspace/multilingual-stt/data/common_voice/ \
        --endpoint-url "$HETZNER_S3_ENDPOINT" 2>/dev/null || true
    aws s3 sync "s3://${HETZNER_S3_BUCKET}/multilingual-stt-general/checkpoints/" \
        /workspace/multilingual-stt/models/whisper-uk-lora/ \
        --endpoint-url "$HETZNER_S3_ENDPOINT" 2>/dev/null || true
    echo "awscli sync done!"
fi

# =============================================
# Git repo
# =============================================
REPO_DIR="/workspace/multilingual-stt"

if [ "$MOUNTED" = true ]; then
    STORAGE_BASE="$MOUNT_POINT/multilingual-stt general"
    REPO_ON_STORAGE="$STORAGE_BASE/multilingual-stt"
    mkdir -p "$STORAGE_BASE"
    if [ -d "$REPO_ON_STORAGE/.git" ]; then
        echo "Repo on storage, pulling..."
        cd "$REPO_ON_STORAGE" && git pull || true
    else
        echo "Cloning repo to storage..."
        git clone https://github.com/BlackVarmir/multilingual-stt.git "$REPO_ON_STORAGE"
    fi
    ln -sfn "$REPO_ON_STORAGE" "$REPO_DIR"
    ln -sfn "$STORAGE_BASE" /workspace/multilingual-stt-general
else
    if [ -d "$REPO_DIR/.git" ]; then
        cd "$REPO_DIR" && git pull || true
    else
        git clone https://github.com/BlackVarmir/multilingual-stt.git "$REPO_DIR"
    fi
fi

# =============================================
# Статус
# =============================================
echo ""
echo "============================================"
echo "  Storage Setup Complete!"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
if [ "$MOUNTED" = true ]; then
    echo "Storage: mounted at $MOUNT_POINT"
else
    echo "Storage: awscli sync (local)"
fi
echo "Repo: $REPO_DIR"
echo ""
echo "  cd /workspace/multilingual-stt"
echo "  python src/asr/train_whisper.py"
echo "============================================"
