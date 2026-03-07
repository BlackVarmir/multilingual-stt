#!/bin/bash
# Налаштування середовища на RunPod A100
echo "=== Setting up RunPod environment ==="

cd /workspace

# Клонуємо репозиторій
if [ ! -d "multilingual-stt" ]; then
    git clone https://github.com/BlackVarmir/multilingual-stt.git
fi
cd multilingual-stt

# Встановлюємо залежності
pip install -r requirements.txt
pip install datasets evaluate jiwer

# Логін в HuggingFace (потрібен для Common Voice)
echo ""
echo "=== HuggingFace Login ==="
echo "You need a HuggingFace token to download Common Voice."
echo "Get it at: https://huggingface.co/settings/tokens"
echo "Then accept the dataset license at:"
echo "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
echo ""
huggingface-cli login

# Завантаження датасетів
echo "=== Downloading datasets ==="
python data/download_datasets.py

echo ""
echo "=== Setup complete! ==="
echo "To start training run:"
echo "  python src/asr/train.py"
echo ""
echo "REMEMBER: Stop the Pod when done!"