#!/bin/bash
set -e

echo "============================================"
echo "  Whisper LoRA Training Pod — Starting..."
echo "============================================"

# --- 1. Credentials ---
if [ -z "$STORAGE_USER" ] || [ -z "$STORAGE_PASS" ]; then
    echo "ERROR: STORAGE_USER and STORAGE_PASS must be set!"
    exit 1
fi

STORAGE_HOST="${STORAGE_HOST:-${STORAGE_USER}.your-storagebox.de}"
STORAGE_PATH="${STORAGE_PATH:-multilingual-stt-general}"
MOUNT_POINT="/workspace/storage"

echo "Storage: //${STORAGE_HOST}/backup/${STORAGE_PATH}"

# --- 2. Mount Storage Box via CIFS ---
apt-get update -qq && apt-get install -y -qq cifs-utils > /dev/null 2>&1

mkdir -p "$MOUNT_POINT"

echo "Mounting Storage Box..."
mount -t cifs "//${STORAGE_HOST}/backup/${STORAGE_PATH}" "$MOUNT_POINT" \
    -o "user=${STORAGE_USER},pass=${STORAGE_PASS},uid=0,gid=0,file_mode=0777,dir_mode=0777"

if mountpoint -q "$MOUNT_POINT"; then
    echo "Storage Box mounted successfully!"
else
    echo "ERROR: CIFS mount failed! Falling back to awscli sync..."

    # Fallback: awscli sync якщо є S3 credentials
    if [ -n "$HETZNER_S3_ACCESS_KEY" ] && [ -n "$HETZNER_S3_SECRET_KEY" ]; then
        echo "Using awscli sync fallback..."
        HETZNER_S3_ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
        HETZNER_S3_BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"

        mkdir -p ~/.aws
        cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = ${HETZNER_S3_ACCESS_KEY}
aws_secret_access_key = ${HETZNER_S3_SECRET_KEY}
EOF
        export AWS_ENDPOINT_URL="$HETZNER_S3_ENDPOINT"
        S3_BASE="s3://${HETZNER_S3_BUCKET}/multilingual-stt-general"

        mkdir -p /workspace/multilingual-stt/data/common_voice
        mkdir -p /workspace/multilingual-stt/models/whisper-uk-lora

        aws s3 sync "${S3_BASE}/CommonVoice/" /workspace/multilingual-stt/data/common_voice/ \
            --endpoint-url "$HETZNER_S3_ENDPOINT" 2>/dev/null || true
        aws s3 sync "${S3_BASE}/checkpoints/" /workspace/multilingual-stt/models/whisper-uk-lora/ \
            --endpoint-url "$HETZNER_S3_ENDPOINT" 2>/dev/null || true

        # Хелпер для upload
        cat > /usr/local/bin/s3-upload << 'SCRIPT'
#!/bin/bash
ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"
echo "Uploading checkpoints to S3..."
aws s3 sync /workspace/multilingual-stt/models/whisper-uk-lora/ \
    "s3://${BUCKET}/multilingual-stt-general/checkpoints/" \
    --endpoint-url "$ENDPOINT"
echo "Done!"
SCRIPT
        chmod +x /usr/local/bin/s3-upload
    else
        echo "WARNING: No S3 credentials either. Working without persistent storage."
    fi
fi

# --- 3. Git repo ---
REPO_DIR="/workspace/multilingual-stt"

# Якщо CIFS змонтований — repo на storage
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    REPO_DIR_STORAGE="$MOUNT_POINT/multilingual-stt"

    if [ -d "$REPO_DIR_STORAGE/.git" ]; then
        echo "Repo exists on storage, pulling latest..."
        cd "$REPO_DIR_STORAGE" && git pull || echo "WARNING: git pull failed"
    else
        echo "Cloning repo to storage..."
        git clone https://github.com/BlackVarmir/multilingual-stt.git "$REPO_DIR_STORAGE"
    fi

    ln -sfn "$REPO_DIR_STORAGE" "$REPO_DIR"
    ln -sfn "$MOUNT_POINT" /workspace/multilingual-stt-general
else
    # Fallback — repo локально
    if [ -d "$REPO_DIR/.git" ]; then
        cd "$REPO_DIR" && git pull || echo "WARNING: git pull failed"
    else
        git clone https://github.com/BlackVarmir/multilingual-stt.git "$REPO_DIR"
    fi
fi

# --- 4. Статус ---
echo ""
echo "============================================"
echo "  Pod Ready!"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"

if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "Storage: CIFS mounted at $MOUNT_POINT"
    echo "Files: $(ls "$MOUNT_POINT" 2>/dev/null | wc -l) items"
else
    echo "Storage: awscli sync (local disk)"
fi

echo ""
echo "Usage:"
echo "  cd /workspace/multilingual-stt"
echo "  python src/asr/train_whisper.py"
echo "============================================"

# --- 5. Keep alive ---
sleep infinity
