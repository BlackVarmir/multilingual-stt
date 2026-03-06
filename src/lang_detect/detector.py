"""Визначення мови аудіо через MMS LID"""

import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from src.config import DEVICE, SAMPLE_RATE


class AudioLanguageDetector:
  """Детектор мови на основі MMS LID-126"""

  def __init__(self, device=DEVICE):
      self.device = device
      print("Loading Language ID model...")
      model_name = "facebook/mms-lid-126"
      self.processor = AutoFeatureExtractor.from_pretrained(model_name)
      self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
      self.model.to(device)
      self.model.eval()
      print("Language ID model loaded")

  def detect(self, audio_chunk):
      """Визначити мову аудіо.

      Повертає: (мова, confidence) — наприклад ("ukr", 0.95)
      """
      inputs = self.processor(
          audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
      )
      input_values = inputs.input_values.to(self.device)

      with torch.no_grad():
          logits = self.model(input_values).logits

      probs = torch.softmax(logits, dim=-1)
      top_prob, top_idx = probs.topk(1)

      lang_id = top_idx.item()
      confidence = top_prob.item()
      lang_code = self.model.config.id2label[lang_id]

      return lang_code, confidence

  def detect_top_n(self, audio_chunk, n=3):
      """Повертає top-N мов з ймовірностями"""
      inputs = self.processor(
          audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
      )
      input_values = inputs.input_values.to(self.device)

      with torch.no_grad():
          logits = self.model(input_values).logits

      probs = torch.softmax(logits, dim=-1)
      top_probs, top_idxs = probs.topk(n)

      results = []
      for i in range(n):
          lang_id = top_idxs[0][i].item()
          confidence = top_probs[0][i].item()
          lang_code = self.model.config.id2label[lang_id]
          results.append((lang_code, confidence))

      return results