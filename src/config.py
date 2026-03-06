"""Конфігурація проекту Multilingual STT"""

import torch

# === Аудіо параметри ===
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # секунди
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 8000 семплів
OVERLAP = 0.1  # 10% overlap між чанками
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160

# === Моделі ===
ASR_MODEL = "facebook/mms-1b-all"
LID_MODEL = "facebook/mms-lid-126"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# === Мови ===
DEFAULT_SOURCE_LANG = "ukr"
DEFAULT_TARGET_LANG = "ukr"
SUPPORTED_LANGUAGES = {
    "uk": {"mms_code": "ukr", "nllb_code": "ukr_Cyrl", "name": "Українська"},
    "ru": {"mms_code": "rus", "nllb_code": "rus_Cyrl", "name": "Русский"},
    "en": {"mms_code": "eng", "nllb_code": "eng_Latn", "name": "English"},
    "de": {"mms_code": "deu", "nllb_code": "deu_Latn", "name": "Deutsch"},
    "pl": {"mms_code": "pol", "nllb_code": "pol_Latn", "name": "Polski"},
}

# === Inference ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True if DEVICE == "cuda" else False

# === VAD ===
VAD_AGGRESSIVENESS = 2  # 0-3, вище = агресивніше фільтрує тишу
VAD_MIN_VOICED_FRAMES = 2  # було 3, тепер швидше стартує. мінімум voiced frames для старту
VAD_MIN_SILENCE_FRAMES = 6  # було 10, тепер швидше фіксує кінець фрази. мінімум silence frames для кінця