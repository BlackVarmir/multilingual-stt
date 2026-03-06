"""MMS ASR модель для розпізнавання мовлення"""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, AutoProcessor
from src.config import ASR_MODEL, DEVICE, USE_FP16, SAMPLE_RATE


class ASRModel:
  """Обгортка над MMS моделлю для розпізнавання мовлення"""

  def __init__(self, model_name=ASR_MODEL, lang="ukr", device=DEVICE):
      self.device = device
      print(f"Loading ASR model: {model_name}...")

      self.processor = AutoProcessor.from_pretrained(model_name)
      self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

      # Встановлюємо мову
      self.set_language(lang)

      # FP16 для GPU
      if USE_FP16 and device == "cuda":
          self.model = self.model.half()

      self.model.to(device)
      self.model.eval()
      print(f"ASR model loaded on {device}")

  def set_language(self, lang_code):
      """Змінити мову розпізнавання (завантажує адаптер ~2MB)"""
      self.processor.tokenizer.set_target_lang(lang_code)
      self.model.load_adapter(lang_code)
      print(f"Language set to: {lang_code}")

  def transcribe(self, audio):
      """Розпізнати мовлення з аудіо.

      audio: numpy array, float32, 16kHz
      Повертає: розпізнаний текст
      """
      # Підготовка вхідних даних
      inputs = self.processor(
          audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
      )
      input_values = inputs.input_values.to(self.device)

      # Inference
      with torch.no_grad():
          logits = self.model(input_values).logits

      # Декодування (greedy — argmax)
      predicted_ids = torch.argmax(logits, dim=-1)
      text = self.processor.batch_decode(predicted_ids)[0]

      return text.strip()