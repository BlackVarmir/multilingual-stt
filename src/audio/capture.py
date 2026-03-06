"""Захоплення аудіо з мікрофону"""

import queue
import numpy as np
import sounddevice as sd
from src.config import SAMPLE_RATE, CHUNK_SIZE


class AudioStream:
  """Потокове захоплення аудіо з мікрофону"""

  def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
      self.sample_rate = sample_rate
      self.chunk_size = chunk_size
      self._queue = queue.Queue()
      self._stream = None

  def _callback(self, indata, frames, time, status):
      """Callback який викликає sounddevice для кожного блоку аудіо"""
      if status:
          print(f"Audio status: {status}")
      # Копіюємо дані в чергу (mono, float32)
      self._queue.put(indata[:, 0].copy())

  def start(self):
      """Почати запис з мікрофону"""
      self._stream = sd.InputStream(
          samplerate=self.sample_rate,
          channels=1,
          dtype=np.float32,
          blocksize=self.chunk_size,
          callback=self._callback,
      )
      self._stream.start()
      print(f"Recording started (sample rate: {self.sample_rate})")

  def stop(self):
      """Зупинити запис"""
      if self._stream:
          self._stream.stop()
          self._stream.close()
          self._stream = None
          print("Recording stopped")

  def get_chunk(self, timeout=1.0):
      """Отримати наступний чанк аудіо з черги"""
      try:
          return self._queue.get(timeout=timeout)
      except queue.Empty:
          return None