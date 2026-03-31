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

# Налаштувати AWS CLI для Hetzner S3
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = ${HETZNER_S3_ACCESS_KEY}
aws_secret_access_key = ${HETZNER_S3_SECRET_KEY}
EOF

cat > ~/.aws/config << EOF
[default]
region = fsn1
s3 =
    endpoint_url = ${HETZNER_S3_ENDPOINT}
EOF

export AWS_ENDPOINT_URL="$HETZNER_S3_ENDPOINT"
S3_BASE="s3://${HETZNER_S3_BUCKET}/multilingual-stt-general"

# --- 2. Перевірка з'єднання з S3 ---
echo "Checking S3 connection..."
if aws s3 ls "s3://${HETZNER_S3_BUCKET}/" --endpoint-url "$HETZNER_S3_ENDPOINT" > /dev/null 2>&1; then
    echo "S3 connection OK!"
else
    echo "ERROR: Cannot connect to S3! Check credentials and endpoint."
    exit 1
fi

# --- 3. Git repo ---
REPO_DIR="/workspace/multilingual-stt"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR" && git pull || echo "WARNING: git pull failed, using existing version"
else
    echo "Cloning repo..."
    git clone https://github.com/BlackVarmir/multilingual-stt.git "$REPO_DIR"
fi

# --- 4. Sync Common Voice з S3 ---
CV_LOCAL="/workspace/multilingual-stt/data/common_voice"
mkdir -p "$CV_LOCAL"

echo "Checking CommonVoice on S3..."
CV_COUNT=$(aws s3 ls "${S3_BASE}/CommonVoice/" --endpoint-url "$HETZNER_S3_ENDPOINT" 2>/dev/null | wc -l)
if [ "$CV_COUNT" -gt 0 ]; then
    echo "Syncing CommonVoice from S3 (archive only, no extract)..."
    aws s3 sync "${S3_BASE}/CommonVoice/" "$CV_LOCAL/" --endpoint-url "$HETZNER_S3_ENDPOINT"
    echo "CommonVoice synced: $(ls "$CV_LOCAL" | wc -l) files"
else
    echo "No CommonVoice data on S3 yet."
fi

# --- 5. Sync existing checkpoints from S3 ---
MODELS_DIR="/workspace/multilingual-stt/models/whisper-uk-lora"
mkdir -p "$MODELS_DIR"

echo "Checking checkpoints on S3..."
CP_COUNT=$(aws s3 ls "${S3_BASE}/checkpoints/" --endpoint-url "$HETZNER_S3_ENDPOINT" 2>/dev/null | wc -l)
if [ "$CP_COUNT" -gt 0 ]; then
    echo "Syncing checkpoints from S3..."
    aws s3 sync "${S3_BASE}/checkpoints/" "$MODELS_DIR/" --endpoint-url "$HETZNER_S3_ENDPOINT"
    echo "Checkpoints synced!"
fi

# --- 6. Створити хелпер скрипти ---
cat > /usr/local/bin/s3-upload << 'SCRIPT'
#!/bin/bash
# Завантажити чекпоінти на S3
ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"
echo "Uploading checkpoints to S3..."
aws s3 sync /workspace/multilingual-stt/models/whisper-uk-lora/ \
    "s3://${BUCKET}/multilingual-stt-general/checkpoints/" \
    --endpoint-url "$ENDPOINT"
echo "Upload complete!"
SCRIPT
chmod +x /usr/local/bin/s3-upload

cat > /usr/local/bin/s3-upload-cv << 'SCRIPT'
#!/bin/bash
# Завантажити Common Voice архів на S3
ENDPOINT="${HETZNER_S3_ENDPOINT:-https://fsn1.your-objectstorage.com}"
BUCKET="${HETZNER_S3_BUCKET:-multilingual-stt}"
echo "Uploading CommonVoice to S3..."
aws s3 sync /workspace/multilingual-stt/data/common_voice/ \
    "s3://${BUCKET}/multilingual-stt-general/CommonVoice/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*clips/*" --exclude "*/clips/*"
echo "Upload complete!"
SCRIPT
chmod +x /usr/local/bin/s3-upload-cv

# --- 7. Статус ---
echo ""
echo "============================================"
echo "  Pod Ready!"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Repo: $REPO_DIR"
echo "S3 Bucket: $HETZNER_S3_BUCKET"
echo ""
echo "Commands:"
echo "  cd /workspace/multilingual-stt"
echo "  python src/asr/train_whisper.py   # start training"
echo "  s3-upload                          # upload checkpoints to S3"
echo "  s3-upload-cv                       # upload CV archive to S3"
echo "============================================"

# --- 8. Keep alive ---
sleep infinity
