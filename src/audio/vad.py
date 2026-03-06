"""Voice Activity Detection через webrtcvad"""

import struct
import numpy as np
import webrtcvad
from src.config import (
  SAMPLE_RATE,
  VAD_AGGRESSIVENESS,
  VAD_MIN_VOICED_FRAMES,
  VAD_MIN_SILENCE_FRAMES,
)


class VoiceActivityDetector:
  """Детектор голосової активності"""

  def __init__(self, sample_rate=SAMPLE_RATE, aggressiveness=VAD_AGGRESSIVENESS):
      self.vad = webrtcvad.Vad(aggressiveness)
      self.sample_rate = sample_rate
      # webrtcvad працює з фреймами 10, 20 або 30 мс
      self.frame_duration_ms = 30
      self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
      self._voiced_count = 0
      self._silence_count = 0
      self._is_speaking = False

  def _float_to_pcm16(self, audio_float):
      """Конвертує float32 [-1,1] в int16 PCM (webrtcvad потребує PCM)"""
      audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
      return struct.pack(f"{len(audio_int16)}h", *audio_int16)

  def is_speech(self, audio_chunk):
      """Перевіряє чи містить чанк мовлення"""
      # Розбиваємо на фрейми по 30мс
      num_frames = len(audio_chunk) // self.frame_size
      if num_frames == 0:
          return False

      voiced_frames = 0
      for i in range(num_frames):
          start = i * self.frame_size
          frame = audio_chunk[start:start + self.frame_size]
          pcm_frame = self._float_to_pcm16(frame)
          if self.vad.is_speech(pcm_frame, self.sample_rate):
              voiced_frames += 1

      return voiced_frames > num_frames * 0.5

  def update(self, audio_chunk):
      """Оновлює стан VAD і повертає (is_speaking, just_started, just_ended)"""
      has_speech = self.is_speech(audio_chunk)

      just_started = False
      just_ended = False

      if has_speech:
          self._voiced_count += 1
          self._silence_count = 0
          if not self._is_speaking and self._voiced_count >= VAD_MIN_VOICED_FRAMES:
              self._is_speaking = True
              just_started = True
      else:
          self._silence_count += 1
          self._voiced_count = 0
          if self._is_speaking and self._silence_count >= VAD_MIN_SILENCE_FRAMES:
              self._is_speaking = False
              just_ended = True

      return self._is_speaking, just_started, just_ended