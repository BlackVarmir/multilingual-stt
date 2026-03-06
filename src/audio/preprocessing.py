"""Препроцесинг аудіо для MMS моделі"""

import numpy as np


def preprocess_audio(audio_chunk, sample_rate=16000):
  """Нормалізує аудіо чанк для подачі в MMS модель.

  MMS сам обробляє raw waveform через Wav2Vec2FeatureExtractor,
  тому тут тільки базова нормалізація.
  """
  # float32, діапазон [-1, 1]
  audio = audio_chunk.astype(np.float32)

  # Нормалізація амплітуди
  max_val = np.abs(audio).max()
  if max_val > 0:
      audio = audio / max_val

  return audio