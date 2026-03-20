# 🎤 Multilingual Speech-to-Text System — Claude Code Master Instruction

> **ВАЖЛИВО ДЛЯ CLAUDE CODE:** Ти — наставник і тренер. Ти НЕ виконуєш команди сам.
> Замість цього ти:
> 1. Пояснюєш користувачу що потрібно зробити на цьому кроці і НАВІЩО
> 2. Даєш точну команду або код який КОРИСТУВАЧ має виконати сам
> 3. Питаєш: "Виконай це і скажи мені результат. Що вивело в консолі?"
> 4. Чекаєш відповіді. Якщо помилка — допомагаєш розібратись
> 5. Тільки після підтвердження "зробив, все ок" → переходиш до наступного кроку
> 
> НІКОЛИ:
> - Не переходь до наступного кроку без підтвердження від користувача
> - Не виконуй нічого сам — давай інструкції користувачу
> - Не давай більше 1-2 команд за раз — чекай підтвердження
> - Не припускай що крок виконаний — завжди питай
> 
> ЯКЩО ПОМИЛКА:
> - Попроси користувача скинути текст помилки
> - Поясни що пішло не так простою мовою
> - Дай виправлену команду
> - Знову чекай підтвердження
> 
> МОВА: Спілкуйся українською, команди та код — англійською.

---

## 📋 Резюме проекту

**Мета:** Real-time streaming система розпізнавання мовлення з автоматичним перекладом. Говориш будь-якою мовою → отримуєш текст на обраній мові.

**Ключові рішення:**

| Параметр | Рішення |
|----------|---------|
| ASR модель | MMS-1B-all CTC (Meta, 1100+ мов) |
| Мовна модель | KenLM 5-gram (Wikipedia + Common Voice) |
| Translation | NLLB-200-distilled-600M (Meta, 200 мов) |
| Language ID | MMS LID-126 |
| Мінімальні мови | Українська, Російська, Англійська |
| Додаткові мови | Будь-які через NLLB (німецька, польська тощо) |
| Streaming | Обов'язково, real-time |
| Inference | GPU (PyTorch FP16) + CPU (ONNX INT8 + CTranslate2) |
| Тренування | RunPod On-Demand, A100 80GB |
| Розробка/тести | Ноутбук, Nvidia Quadro P1000 (4GB VRAM, $0) |
| Деплой (production) | VPS, CPU-only (ONNX INT8 + CTranslate2) |
| Підхід | MVP → поступове покращення |
| Декодер | Greedy CTC + Beam Search з KenLM (pyctcdecode) |
| Two-pass | Pass 1 — швидкий CTC, Pass 2 — async post-processing |

---

## 🖥️ ЧАСТИНА 0 — Налаштування RunPod GPU Cloud

> Це робиш ТИ (людина), не Claude Code. Ця секція — покрокова інструкція для тебе.

### Крок 0.1 — Реєстрація на RunPod

```
1. Відкрий https://www.runpod.io
2. Натисни "Get Started" / "Sign Up"
3. Зареєструйся (email або GitHub)
4. Підтверди email
```

**✅ Підтвердження:** Ти зареєстрований і бачиш Dashboard?

### Крок 0.2 — Поповнення балансу

```
1. Перейди в Settings → Billing
2. Додай спосіб оплати (картка Visa/Mastercard)
3. Поповни баланс — рекомендую $25 для початку
   (цього вистачить на ~16 годин A100)
4. Увімкни "Auto-recharge" якщо хочеш автопоповнення
```

**✅ Підтвердження:** Баланс поповнений?

### Крок 0.3 — Створення Network Volume (постійне сховище)

```
1. Перейди в Storage → Network Volumes
2. Натисни "Create Network Volume"
3. Налаштування:
   - Name: "stt-project"
   - Region: EU-RO-1 (Румунія — найближче до України) 
     або US-TX-3 (якщо EU немає)
   - Size: 20 GB (достатньо для моделей + датасетів)
4. Натисни "Create"
```

**Вартість:** ~$1.40/міс за 20GB (платиш навіть коли Pod вимкнений).
**Навіщо:** Цей диск зберігається між Pod'ами. Натренував модель → зберіг сюди → видалив Pod → модель на місці.

**✅ Підтвердження:** Network Volume створений?

### Крок 0.4 — Налаштування ноутбука (розробка тут)

```
Вся розробка ведеться на твоєму ноутбуці з Quadro P1000.
RunPod потрібен ТІЛЬКИ для тренування (Фаза 4).

1. Переконайся що встановлено:
   - Python 3.10+
   - CUDA Toolkit (для P1000)
   - Git

2. Перевір GPU:
   nvidia-smi
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
   
   Маєш побачити Quadro P1000 і CUDA available = True.

3. Налаштуй Git:
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
```

**✅ Підтвердження:** Python, CUDA і Git працюють на ноутбуці?

### Крок 0.5 — Створення GitHub репозиторію

```
1. Створи новий репозиторій на GitHub: "multilingual-stt"
2. Клонуй на ноутбук:
   git clone https://github.com/YOUR_USERNAME/multilingual-stt.git
   cd multilingual-stt
```

**✅ Підтвердження:** Репозиторій створений і клонований?

### Крок 0.6 — Пам'ятка: коли знадобиться RunPod

```
RunPod тобі знадобиться тільки на Фазі 4 (Fine-tune).
Коли прийде час, ось що робити:

1. Перейди на RunPod → Pods → Deploy
2. GPU: A100 SXM або A100 PCIe (80GB)
3. Secure Cloud
4. Template: "RunPod PyTorch 2.4.0"
5. Container Disk: 20 GB
6. Підключи Network Volume: "stt-project"
   - Mount Path: /workspace/storage
7. Expose Ports: 8888 (Jupyter), 22 (SSH)
8. Deploy → зачекай 1-2 хвилини

Підключення до Pod:
  A) Web Terminal: Connect → Start Web Terminal
  B) SSH: Connect → скопіюй SSH команду

Після тренування:
  - Зберіг модель на /workspace/storage
  - git push
  - ВИДАЛИВ Pod (Terminate)!
  - Забрав модель на ноутбук

⚠️ НЕ ЗАБУДЬ ВИДАЛИТИ POD ПІСЛЯ ТРЕНУВАННЯ!
```

Цей крок — тільки для ознайомлення. Повернешся сюди на Фазі 4.

---

> **ТЕПЕР — переходь до Claude Code.**
> Відкрий термінал на ноутбуці, зайди в папку проекту і запусти Claude Code.
> Дай йому цю інструкцію. Він буде вести тебе по кроках, починаючи з Фази 1.

---

## 🏗️ ФАЗА 1 — MVP (Мінімальний робочий продукт)

**Мета:** Працюючий streaming STT для української. Без перекладу, без C++.
**Час:** ~1-2 тижні
**Де:** На ноутбуці (P1000) або навіть CPU

### Крок 1.1 — Ініціалізація проекту

> Claude Code: поясни що потрібно зробити і дай команди по одній.

**Що кажеш користувачу:**

"Зараз ми створимо структуру проекту. Виконай цю команду в терміналі на ноутбуці, в папці проекту:"

```bash
mkdir -p multilingual-stt/{src/{audio,asr,decoder,lang_detect,translation,abbreviations,postprocessing},data,scripts,models,tests}
cd multilingual-stt
```

"Виконав? Що показує `ls`? Скинь результат."

Після підтвердження — дай наступну команду:

```bash
touch src/__init__.py src/audio/__init__.py src/asr/__init__.py src/decoder/__init__.py
touch src/config.py src/pipeline.py main.py
```

"Виконав? Тепер створимо requirements.txt. Відкрий файл і вставь цей вміст:"

```
# Скажи користувачу вставити requirements.txt вміст (той самий що нижче)
```

```
Створи структуру проекту:

multilingual-stt/
├── src/
│   ├── __init__.py
│   ├── config.py              # Конфігурація всього проекту
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py         # Захоплення аудіо з мікрофону
│   │   ├── preprocessing.py   # Mel-spectrogram
│   │   └── vad.py             # Voice Activity Detection
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── model.py           # MMS ASR модель
│   │   └── streaming.py       # Streaming inference
│   ├── decoder/
│   │   ├── __init__.py
│   │   └── greedy.py          # Greedy CTC decoder
│   └── pipeline.py            # Головний streaming pipeline
├── data/
│   ├── download_datasets.py   # Завантаження датасетів
│   └── prepare_dataset.py     # Підготовка даних
├── scripts/
│   ├── setup_runpod.sh        # Налаштування середовища на RunPod
│   ├── download_models.sh     # Завантаження моделей з HuggingFace
│   └── export_model.py        # Експорт моделі для CPU inference
├── models/                    # Збережені моделі
├── tests/
│   ├── test_audio.py
│   ├── test_asr.py
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
├── Dockerfile                 # Для деплою
└── README.md

Та requirements.txt:

# Core ML
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.36.0

# Audio
librosa>=0.10.0
sounddevice>=0.4.6
webrtcvad>=2.0.10
soundfile>=0.12.0

# MMS & NLLB
sentencepiece>=0.1.99
protobuf>=3.20.0

# Data
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0

# CPU Inference (для пізніших фаз)
onnxruntime>=1.16.0
ctranslate2>=4.0.0

# Utils
tqdm>=4.65.0
PyYAML>=6.0

# Testing
pytest>=7.4.0
```

**✅ Запитай користувача:** "Виконав усі команди? Покажи що вивело `ls -la`. Все створилось?"

### Крок 1.2 — Конфігурація

```
Створи src/config.py:

Параметри:
- SAMPLE_RATE = 16000
- CHUNK_DURATION = 0.5  # секунди
- CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # 8000 семплів
- OVERLAP = 0.1  # 10% overlap
- N_MELS = 80
- N_FFT = 512
- HOP_LENGTH = 160

Моделі:
- ASR_MODEL = "facebook/mms-1b-all"  # MMS з підтримкою 1100+ мов
  (або "facebook/mms-300m" якщо потрібна менша модель)
- LID_MODEL = "facebook/mms-lid-126"  # Language ID
- TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"  # Translation

Мови:
- DEFAULT_SOURCE_LANG = "ukr"  # ISO 639-3 для MMS
- DEFAULT_TARGET_LANG = "ukr"
- SUPPORTED_LANGUAGES = {
    "uk": {"mms_code": "ukr", "nllb_code": "ukr_Cyrl", "name": "Українська"},
    "ru": {"mms_code": "rus", "nllb_code": "rus_Cyrl", "name": "Русский"},
    "en": {"mms_code": "eng", "nllb_code": "eng_Latn", "name": "English"},
    "de": {"mms_code": "deu", "nllb_code": "deu_Latn", "name": "Deutsch"},
    "pl": {"mms_code": "pol", "nllb_code": "pol_Latn", "name": "Polski"},
    # Додавай мови за потребою
  }

Inference:
- DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
- USE_FP16 = True if DEVICE == "cuda" else False
```

**✅ Запитай користувача:** "Створи файл src/config.py і вставь туди цей код. Зроби і скажи коли готово."

### Крок 1.3 — Audio Capture та Preprocessing

```
Створи src/audio/capture.py:
- Клас AudioStream:
  - Захоплення аудіо з мікрофону через sounddevice
  - Callback-based streaming (не блокуючий)
  - Queue для буферизації чанків
  - Методи: start(), stop(), get_chunk()

Створи src/audio/preprocessing.py:
- Функція compute_features(audio_chunk, sr):
  - Для MMS: повертає raw waveform як torch.Tensor
    (MMS сам обробляє аудіо через Wav2Vec2FeatureExtractor)
  - Нормалізація: float32, [-1, 1]

Створи src/audio/vad.py:
- Клас VoiceActivityDetector:
  - webrtcvad з aggressiveness = 2
  - is_speech(chunk) → bool
  - Буфер: min 3 voiced frames для старту, min 10 unvoiced для кінця
```

**✅ Запитай користувача:** "Створи ці три файли по одному. Кажи коли кожен готовий, я дам наступний."

### Крок 1.4 — MMS ASR модель

```
Створи src/asr/model.py:
- Клас ASRModel:
  - __init__(self, lang="ukr", device="cuda"):
    - Завантажує MMS модель:
      from transformers import Wav2Vec2ForCTC, AutoProcessor
      self.processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
      self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    - Встановлює мову:
      self.processor.tokenizer.set_target_lang(lang)
      self.model.load_adapter(lang)
    - FP16 якщо GPU: self.model = self.model.half()
    - Переміщує на device
    
  - set_language(self, lang_code):
    - Динамічна зміна мови (завантажує новий адаптер ~2MB)
    - self.processor.tokenizer.set_target_lang(lang_code)
    - self.model.load_adapter(lang_code)
    
  - transcribe(self, audio_tensor) → str:
    - inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
    - З torch.no_grad(): logits = self.model(**inputs).logits
    - ids = torch.argmax(logits, dim=-1)
    - text = self.processor.batch_decode(ids)
    - Повертає текст
    
  VRAM: ~600MB для MMS-300M, ~2GB для MMS-1B
  Час inference: P1000 ~50-100ms, CPU ~100-200ms для 0.5s чанка
```

**✅ Запитай користувача:** "Створи файл src/asr/model.py з цим кодом. Зроби і скажи коли готово."

### Крок 1.5 — CTC Decoder

```
Створи src/decoder/greedy.py:
- Клас GreedyCTCDecoder:
  - decode(logits, processor) → str:
    - argmax по vocabulary
    - processor.batch_decode()
    - Повертає текст
  - Простий, швидкий, для MVP достатньо
  - Beam search буде в Фазі 5
```

**✅ Запитай користувача:** "Створи файл src/decoder/greedy.py. Кажи коли готово."

### Крок 1.6 — Streaming Pipeline

```
Створи src/pipeline.py:
- Клас StreamingSTTPipeline:
  - __init__(self, source_lang="ukr", target_lang="ukr"):
    - Ініціалізує AudioStream, ASRModel, VAD
    - source_lang — мова на якій говоримо
    - target_lang — мова в яку перекладаємо (для Фази 2)
    
  - process_chunk(audio_chunk) → Optional[str]:
    1. VAD: якщо тиша → None
    2. Processor: audio → input features
    3. ASR: features → logits → text
    4. Повернути текст
    
  - run(callback):
    - Головний цикл
    - Для кожного чанка: result = process_chunk(chunk)
    - callback(result, is_final=False)
    - Коли VAD → кінець мовлення: callback(sentence, is_final=True)
    
  - set_languages(source, target):
    - Змінити мови на льоту
    
  - stop(): зупинити все

Створи main.py:
  - CLI з аргументами: --source-lang, --target-lang, --device
  - Ініціалізація pipeline
  - callback = print() для MVP
  - Graceful shutdown (Ctrl+C)
```

**✅ Запитай користувача:** "Створи src/pipeline.py та main.py з цим кодом. Коли зробиш — запусти `python main.py --help` і покажи результат."

### Крок 1.7 — Тести

```
Створи тести:

tests/test_audio.py:
- Тест mel-spectrogram shape
- Тест VAD на тиші vs мовленні

tests/test_asr.py:
- Тест завантаження моделі
- Тест зміни мови (uk → en → uk)
- Тест inference на тестовому аудіо файлі
- Тест що output — валідний UTF-8

tests/test_pipeline.py:
- End-to-end тест з аудіо файлу (без мікрофону)
- Benchmark: час обробки одного чанка
```

**✅ Запитай користувача:** "Створи тест-файли і запусти `pytest tests/`. Покажи що вивело."

### Крок 1.8 — Скрипти для RunPod

```
Створи scripts/setup_runpod.sh:
#!/bin/bash
# Запускається один раз при першому старті Pod'а
pip install -r requirements.txt
python -c "
from transformers import Wav2Vec2ForCTC, AutoProcessor
# Кешуємо моделі
processor = AutoProcessor.from_pretrained('facebook/mms-1b-all')
model = Wav2Vec2ForCTC.from_pretrained('facebook/mms-1b-all')
print('MMS model downloaded!')
"
echo "Setup complete!"

Створи scripts/download_models.sh:
#!/bin/bash
# Завантажує моделі з HuggingFace в /workspace/storage/models
mkdir -p /workspace/storage/models
python -c "
from transformers import Wav2Vec2ForCTC, AutoProcessor
p = AutoProcessor.from_pretrained('facebook/mms-1b-all')
m = Wav2Vec2ForCTC.from_pretrained('facebook/mms-1b-all')
p.save_pretrained('/workspace/storage/models/mms-1b-all')
m.save_pretrained('/workspace/storage/models/mms-1b-all')
print('Models saved to Network Volume!')
"
```

**✅ Запитай користувача:** "Створи ці скрипти і запусти `bash scripts/setup_runpod.sh`. Покажи результат."

### Критерії завершення Фази 1

- [ ] Pipeline запускається і слухає мікрофон (або файл)
- [ ] VAD коректно детектує мовлення vs тишу
- [ ] MMS видає текст українською
- [ ] Зміна мови працює (uk → en → ru)
- [ ] Streaming: текст з'являється по мірі говоріння
- [ ] Тести проходять

**✅ Запитай користувача:** "Запусти повний pipeline з аудіо файлом. Працює? Текст з'являється? Якщо так — вітаю, Фаза 1 завершена! Готовий до Фази 2?"

---

## 🌍 ФАЗА 2 — Мультимовність та Translation

**Мета:** Автоматичне визначення мови + переклад на обрану мову.
**Час:** ~1-2 тижні
**Де:** Ноутбук (P1000) або CPU

### Крок 2.1 — Language Identification (MMS LID)

```
Створи src/lang_detect/
  ├── __init__.py
  └── detector.py

Клас AudioLanguageDetector:
  - __init__(self):
    - Завантажує MMS LID модель:
      from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
      self.model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-126")
      self.processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")
    - VRAM: ~100MB
    
  - detect(audio_chunk) → Tuple[str, float]:
    - Повертає (мова, confidence)
    - Наприклад: ("ukr", 0.95)
    
  - detect_top_n(audio_chunk, n=3) → List[Tuple[str, float]]:
    - Top-N мов з ймовірностями
    
  Час: ~5-10ms на GPU, ~20-50ms на CPU
```

**✅ Запитай користувача:** "Створи папку src/lang_detect/ і файл detector.py з цим кодом. Кажи коли готово."

### Крок 2.2 — Translation Module (NLLB-200)

```
Створи src/translation/
  ├── __init__.py
  ├── translator.py      # Головний перекладач
  └── models.py          # NLLB wrapper

Клас NLLBTranslator:
  - __init__(self, device):
    - from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    - self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    - self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    - FP16 на GPU
    - VRAM: ~1.2GB
    
  - translate(text, source_lang, target_lang) → str:
    - source_lang / target_lang у форматі NLLB: "ukr_Cyrl", "eng_Latn" тощо
    - self.tokenizer.src_lang = source_lang
    - inputs = self.tokenizer(text, return_tensors="pt")
    - generated = self.model.generate(**inputs, forced_bos_token_id=target_lang_id)
    - Повертає перекладений текст
    
  - translate_batch(texts, source_lang, target_lang) → List[str]:
    - Батч-переклад для ефективності

Клас Translator (фасад):
  - __init__(self, device):
    - Ініціалізує NLLBTranslator
    
  - translate(text, src, tgt) → str:
    - Якщо src == tgt → повернути без перекладу
    - Інакше → NLLB переклад
    - Кешування частих пар
```

**✅ Запитай користувача:** "Створи папку src/translation/ і два файли. Давай по одному. Готовий?"

### Крок 2.3 — Abbreviation Handler

```
Створи src/abbreviations/
  ├── __init__.py
  ├── handler.py
  └── db.json

db.json — база абревіатур (100+ записів):
{
  "STT": {"full": "Speech-to-Text", "uk": "мовлення в текст"},
  "API": {"full": "Application Programming Interface", "uk": "програмний інтерфейс"},
  "GPU": {"full": "Graphics Processing Unit", "uk": "графічний процесор"},
  "AI": {"full": "Artificial Intelligence", "uk": "штучний інтелект"},
  "ML": {"full": "Machine Learning", "uk": "машинне навчання"},
  ...100+ IT абревіатур
}

Клас AbbreviationHandler:
  - detect(text) → List[абревіатури]
  - process(text, action="keep_original") → str
    - "keep_original": залишити як є
    - "expand_uk": розгорнути українською
    - "expand_en": розгорнути англійською
  - add_abbreviation(abbr, full, uk)
```

**✅ Запитай користувача:** "Створи src/abbreviations/db.json та handler.py. Покажи коли зробиш."

### Крок 2.4 — Оновлений Pipeline з перекладом

```
Оновити src/pipeline.py:

StreamingSTTPipeline тепер:
  - __init__(self, source_lang="uk", target_lang="uk"):
    - ASR (MMS)
    - LID (MMS LID)
    - Translator (NLLB)
    - AbbreviationHandler
    
  - process_chunk(audio_chunk) → dict:
    # Pass 1 — швидкий (показати одразу)
    detected_lang = self.lid.detect(audio_chunk)  # ~5ms
    self.asr.set_language(detected_lang)
    raw_text = self.asr.transcribe(audio_chunk)    # ~10ms
    return {"text": raw_text, "lang": detected_lang, "is_final": False}
    
  - finalize_sentence(raw_text, detected_lang) → dict:
    # Pass 2 — async post-processing
    text = self.abbreviations.process(raw_text)
    if detected_lang != self.target_lang:
      text = self.translator.translate(text, detected_lang, self.target_lang)
    return {"text": text, "lang": self.target_lang, "is_final": True}

VRAM бюджет (P1000, 4GB — впритул але влізе):
  MMS-300M:   ~0.6GB  (ВАЖЛИВО: використовуй 300M, не 1B!)
  MMS LID:    ~0.1GB
  NLLB-600M:  ~1.2GB
  Буфери:     ~0.3GB
  Разом:      ~2.2GB з 4GB ✅
  
  ⚠️ MMS-1B (~2GB) НЕ влізе разом з NLLB на P1000!
  Використовуй MMS-300M для розробки на ноутбуці.
  На VPS (CPU) розмір VRAM не має значення — все в RAM.
```

**✅ Запитай користувача:** "Оновити src/pipeline.py цим кодом. Заміни старий вміст. Кажи коли готово."

### Крок 2.5 — Тести мультимовності

```
tests/test_translation.py:
- "Привіт, як справи?" (uk→en) → "Hello, how are you?"
- "Привет, как дела?" (ru→uk) → "Привіт, як справи?"
- Тест абревіатур: "GPU дуже потужний" → зберігає "GPU"
- Тест LID: правильно визначає uk/ru/en

tests/test_multilingual_pipeline.py:
- End-to-end: аудіо файл uk → текст en
- Benchmark: повний pipeline latency
```

**✅ Запитай користувача:** "Створи тест-файли і запусти `pytest tests/`. Покажи що вивело." Фаза 2 завершена?

---

## ⚡ ФАЗА 3 — Оптимізація та CPU Inference

**Мета:** Зробити inference швидким на GPU і працюючим на CPU.
**Час:** ~2-3 тижні
**Де:** A100 (RunPod) для експорту, P1000/CPU для тестів

### Крок 3.1 — Експорт моделей в ONNX

```
Створи scripts/export_model.py:
- Експорт MMS CTC → ONNX:
  torch.onnx.export(model, dummy_input, "models/mms_ctc.onnx",
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={"input_values": {0: "batch", 1: "time"},
                  "logits": {0: "batch", 1: "time"}},
    opset_version=17)

- Експорт NLLB → CTranslate2 (краще для seq2seq):
  import ctranslate2
  ct2_model = ctranslate2.converters.TransformersConverter("facebook/nllb-200-distilled-600M")
  ct2_model.convert("models/nllb_ct2", quantization="int8")

- Валідація: порівняй output ONNX vs PyTorch (tolerance < 1e-3)
```

**✅ Запитай користувача:** "Запусти `python scripts/export_model.py`. Покажи результат і розмір файлів."

### Крок 3.2 — INT8 Квантизація

```
Створи scripts/quantize_model.py:
- MMS ONNX → INT8:
  from onnxruntime.quantization import quantize_dynamic, QuantType
  quantize_dynamic("models/mms_ctc.onnx", "models/mms_ctc_int8.onnx",
    weight_type=QuantType.QInt8)

- NLLB вже квантизується через CTranslate2 (Крок 3.1)

Порівняй розміри:
  MMS FP32:   ~1.2GB → INT8: ~300MB
  NLLB FP32:  ~1.2GB → INT8 CT2: ~300-400MB
  
Порівняй якість:
  Запусти тести з Фази 2 на квантизованих моделях
  WER degradation має бути < 2%
```

**✅ Запитай користувача:** "Запусти `python scripts/quantize_model.py`. Порівняй розміри файлів до і після."

### Крок 3.3 — ONNX Runtime Inference (GPU + CPU)

```
Створи src/asr/inference_onnx.py:
- Клас ONNXASRModel:
  - __init__(self, model_path, device="cpu"):
    - import onnxruntime as ort
    - providers = ["CUDAExecutionProvider"] if device == "cuda" 
                  else ["CPUExecutionProvider"]
    - self.session = ort.InferenceSession(model_path, providers=providers)
    
  - transcribe(audio) → str:
    - inputs = {"input_values": audio_array}
    - logits = self.session.run(None, inputs)[0]
    - decode logits → text
    
  Час (0.5s аудіо):
    GPU (P1000):  ~50-80ms
    CPU (i7-12700K):  ~50-80ms
    CPU (4 vCPU VPS): ~90-140ms

Створи src/translation/inference_ct2.py:
- Клас CT2Translator:
  - __init__(self, model_path, device="cpu"):
    - self.translator = ctranslate2.Translator(model_path, device=device)
    
  - translate(text, src_lang, tgt_lang) → str:
    - tokenize → translate → detokenize
    
  Час:
    GPU:  ~10-20ms
    CPU:  ~80-200ms
```

**✅ Запитай користувача:** "Створи ці файли і запусти тест: `python -c "from src.asr.inference_onnx import ONNXASRModel"`. Працює?"

### Крок 3.4 — C++ Audio Preprocessing (опційно, для максимальної швидкості)

```
Створи src/audio_cpp/:
  CMakeLists.txt
  mel_spectrogram.cpp / .h   — CUDA або CPU mel-spectrogram
  vad.cpp / .h               — Енергетичний VAD
  audio_processor.cpp / .h   — Головний клас
  bindings.cpp               — pybind11

Прискорення: Python librosa ~200ms → C++ ~5ms для mel-spectrogram
Можна пропустити якщо Python варіант достатньо швидкий.
```

**✅ Запитай користувача:** "C++ модуль — опціональний. Хочеш робити зараз чи пропустимо і повернемось пізніше?"

### Крок 3.5 — Unified Pipeline (GPU + CPU)

```
Оновити src/pipeline.py:
- Автоматичний вибір backend:
  if device == "cuda":
    self.asr = ASRModel(...)           # PyTorch, повна модель
    self.translator = NLLBTranslator(...)
  else:
    self.asr = ONNXASRModel(...)       # ONNX INT8
    self.translator = CT2Translator(...) # CTranslate2 INT8

- Benchmark mode:
  pipeline.run(mode="benchmark"):
    100 чанків → mean, p50, p95, p99 latency
    Target GPU: < 30ms total
    Target CPU: < 150ms total
```

**✅ Запитай користувача:** "Оновити src/pipeline.py цим кодом. Заміни старий вміст. Кажи коли готово." Фаза 3 завершена?

---

## 🎯 ФАЗА 4 — Fine-tuning на українських даних

**Мета:** Значно покращити якість розпізнавання.
**GPU:** A100 80GB (RunPod On-Demand)

### Датасети

- **FLEURS Ukrainian** (Google) — початкове fine-tuning (~3k samples)
- **Common Voice Ukrainian v24** (Mozilla) — основне тренування (~27k train, ~10k test, ~10k dev)

### Аугментація (src/asr/augmentation.py)

- add_noise (SNR 10-30dB, 50% шанс)
- speed_perturb (0.9-1.1x через інтерполяцію, 50% шанс)
- SpecAugment (вбудований в Wav2Vec2: mask_time_prob=0.4, mask_feature_prob=0.1)

### Історія тренувань

| Модель | База | Дані | Особливості | Greedy WER | Beam+KenLM |
|--------|------|------|-------------|-----------|------------|
| multilingual-stt-uk | mms-1b-all | FLEURS | frozen encoder, lr=1e-4 | 21.2% | — |
| multilingual-stt-uk-cv2 | uk | Common Voice | frozen, lr=3e-5, 5 epochs | 41.2%* | 19.4% |
| multilingual-stt-uk-cv3 | uk-cv2 | CV + val | frozen, SpecAugment, cosine, lr=1e-5 | 38.2%* | 20.6% |
| multilingual-stt-uk-cv4 | uk-cv3 | CV + val | **unfrozen encoder**, lr=5e-6 | 39.2%* | 21.1% |

*\* Виміряно на 200 random test samples (seed=42). RunPod eval cv4 = 27.87%*

### Ключові уроки

1. **Frozen vs unfrozen feature encoder**: розморожений encoder дає потенційно краще, але потребує дуже низький lr (1e-6) інакше модель розвалюється (catastrophic forgetting після epoch 3-4)
2. **Layer-wise lr**: feature encoder потребує 5-10x менший lr ніж решта моделі
3. **SpecAugment**: вбудований в Wav2Vec2 через config, дає невелике покращення
4. **KenLM**: beam search з KenLM дає 15-20% покращення WER поверх greedy
5. **Проблема `<unk>`**: модель ставить `<unk>` замість першої літери речення (CTC початок аудіо)
6. **Early stopping**: обов'язковий, patience=5. Без нього модель перенавчається

### Наступний крок: cv5

Скрипт готовий (src/asr/train.py):
- База: cv4, layer-wise lr (feature encoder 1e-6, решта 5e-6)
- Cosine scheduler з 10% warmup
- SpecAugment + speed perturbation
- 15 epochs, early stopping patience=5

### Критерії завершення Фази 4

- [x] Fine-tuned модель краще ніж базова MMS
- [x] Моделі на HuggingFace (BlackVarmir/multilingual-stt-uk-cv*)
- [ ] Greedy WER < 15% (поточний найкращий: ~27-39%)
- [ ] Нові ONNX + INT8 моделі експортовані з найкращої версії

---

## 🔧 ФАЗА 5 — Advanced Post-processing

**Мета:** KenLM, пунктуація, перевірка орфографії.
**Де:** Ноутбук або CPU (KenLM будувався в WSL)

### Крок 5.1 — KenLM Language Model (ЗРОБЛЕНО)

Побудовано два KenLM 5-gram:

| Модель | Джерело | Розмір | WER з cv2 |
|--------|---------|--------|-----------|
| uk_5gram.bin | Common Voice (47k речень) | 29MB | 20.3% |
| uk_5gram_wiki.bin | Wikipedia + CV (12M речень) | 8.37GB | **19.4%** |

- Будувалось через KenLM lmplz + build_binary в WSL (потрібен libboost)
- Декодер: src/decoder/beam_search.py (pyctcdecode)
- Alpha=0.5, beta=1.0 (дефолтні — найкращі для wiki моделі)
- Wiki модель не вміщає unigrams в пам'ять на ноутбуці (8.6M слів) — працює без них

### Крок 5.2 — Punctuation & Capitalization (ЗРОБЛЕНО)

src/postprocessing/punctuation.py — працює.

### Крок 5.3 — Spelling Correction (ЗРОБЛЕНО)

src/postprocessing/spelling.py — працює.

---

## 🚀 ФАЗА 6 — Production Ready

**Мета:** CLI, API сервер, Docker, деплой.
**Час:** ~1 тиждень

### Крок 6.1 — CLI інтерфейс

```
Красивий CLI з кольоровим виводом:
- Зелений: розпізнані слова
- Жовтий: перекладені слова
- Синій: абревіатури
- Аргументи: --source-lang, --target-lang, --device, --model-path
```

### Крок 6.2 — WebSocket API сервер

```
FastAPI + WebSocket:
- Client надсилає аудіо chunks
- Server відповідає текстом
- Protocol: JSON {"type": "partial"/"final", "text": "...", "source_lang": "uk", "target_lang": "en"}
```

### Крок 6.3 — Docker

```
Dockerfile для CPU inference:
- Python 3.11 + ONNX Runtime + CTranslate2
- Моделі включені в image (~700MB)
- Запуск: docker run -p 8000:8000 stt-server
- Працює на будь-якому сервері без GPU
```

### Крок 6.4 — Деплой на свій сервер

```
Варіанти:
A) Docker на будь-якому VPS ($5-10/міс)
B) На Windows Server через WSL2 або нативно
C) На Raspberry Pi (повільно але працює)
D) Як мікросервіс у Docker Compose
```

**✅ На кожному підкроці Фази 6 — дай інструкцію, чекай підтвердження, потім наступний крок.

---

## 📊 Реальні результати

| Фаза | WER (greedy) | WER (beam+KenLM) | Статус |
|------|-------------|-------------------|--------|
| 1 MVP (базова MMS-1B) | ~50-60% | — | ✅ |
| 4 Fine-tune FLEURS | 21.2% | — | ✅ |
| 4 Fine-tune + CV (cv2) | 41.2% | 19.4% | ✅ |
| 4 + SpecAugment (cv3) | 38.2% | 20.6% | ✅ |
| 4 + Unfrozen encoder (cv4) | 39.2% | 21.1% | ✅ |
| 5 KenLM beam search | — | 19.4% (wiki) | ✅ |
| 4 + Layer-wise lr (cv5) | ? | ? | Готовий до запуску |

**Витрати на RunPod:** ~$15-20 загалом на всі тренування.

---

## ⚠️ Важливі нотатки

1. **Зберігай прогрес!** git push після кожної сесії. Чекпоінти → Network Volume.
2. **Вимикай Pod!** Закінчив роботу → Stop Pod. Не забувай!
3. **A100 тільки для тренування.** Розробка — на ноутбуці з P1000 ($0).
4. **CPU моделі — окремо.** Після ONNX/CT2 експорту GPU не потрібна для inference.
5. **Тестуй на кожному кроці.** Не переходь до наступної фази без робочих тестів.

---

## 🔗 Корисні посилання

- MMS: https://huggingface.co/facebook/mms-1b-all
- NLLB: https://huggingface.co/facebook/nllb-200-distilled-600M
- MMS LID: https://huggingface.co/facebook/mms-lid-126
- RunPod Docs: https://docs.runpod.io
- ONNX Runtime: https://onnxruntime.ai
- CTranslate2: https://github.com/OpenNMT/CTranslate2
- Common Voice: https://commonvoice.mozilla.org/uk
