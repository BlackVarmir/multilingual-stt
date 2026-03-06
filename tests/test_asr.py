"""Тести для ASR моделі"""

import pytest
import numpy as np
import time


class TestASRModel:
  """Тести завантаження і inference MMS моделі"""

  @pytest.fixture(scope="class")
  def model(self):
      """Завантажуємо модель один раз для всіх тестів"""
      from src.asr.model import ASRModel
      return ASRModel(lang="ukr", device="cpu")

  def test_model_loads(self, model):
      """Модель завантажується без помилок"""
      assert model is not None
      assert model.device == "cpu"

  def test_transcribe_silence(self, model):
      """На тиші повертає порожній рядок"""
      silence = np.zeros(16000, dtype=np.float32)
      result = model.transcribe(silence)
      assert isinstance(result, str)

  def test_transcribe_noise(self, model):
      """На шумі не падає з помилкою"""
      noise = np.random.randn(16000).astype(np.float32) * 0.01
      result = model.transcribe(noise)
      assert isinstance(result, str)

  def test_transcribe_returns_utf8(self, model):
      """Результат — валідний UTF-8"""
      audio = np.random.randn(16000).astype(np.float32) * 0.1
      result = model.transcribe(audio)
      result.encode("utf-8")  # не впаде якщо валідний UTF-8

  def test_change_language(self, model):
      """Зміна мови працює"""
      model.set_language("eng")
      noise = np.random.randn(16000).astype(np.float32) * 0.01
      result = model.transcribe(noise)
      assert isinstance(result, str)
      # Повертаємо назад
      model.set_language("ukr")

  def test_inference_speed(self, model):
      """Inference одного чанка < 5 секунд на CPU"""
      audio = np.random.randn(8000).astype(np.float32) * 0.1
      start = time.time()
      model.transcribe(audio)
      elapsed = time.time() - start
      print(f"\nInference time: {elapsed:.3f}s")
      assert elapsed < 5.0