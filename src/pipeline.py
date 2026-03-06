"""Головний streaming pipeline для розпізнавання мовлення"""

import numpy as np
from src.config import SAMPLE_RATE, SUPPORTED_LANGUAGES
from src.audio.capture import AudioStream
from src.audio.vad import VoiceActivityDetector
from src.audio.preprocessing import preprocess_audio
from src.asr.model import ASRModel


class StreamingSTTPipeline:
  """Streaming Speech-to-Text Pipeline"""

  def __init__(self, source_lang="uk", device="cpu"):
      lang_config = SUPPORTED_LANGUAGES[source_lang]
      mms_code = lang_config["mms_code"]

      print("Initializing pipeline...")
      self.audio_stream = AudioStream()
      self.vad = VoiceActivityDetector()
      self.asr = ASRModel(lang=mms_code, device=device)
      self.source_lang = source_lang
      self._running = False
      # Буфер для збору фрагментів мовлення
      self._speech_buffer = []
      print("Pipeline ready!")

  def process_chunk(self, audio_chunk):
      """Обробити один чанк аудіо.

      Повертає dict з результатом або None якщо тиша.
      """
      is_speaking, just_started, just_ended = self.vad.update(audio_chunk)

      if just_started:
          self._speech_buffer = []

      if is_speaking:
          self._speech_buffer.append(audio_chunk)
          # Розпізнаємо поточний буфер (partial result)
          audio = np.concatenate(self._speech_buffer)
          audio = preprocess_audio(audio)
          text = self.asr.transcribe(audio)
          return {"text": text, "lang": self.source_lang, "is_final": False}

      if just_ended and self._speech_buffer:
          # Кінець фрази — фінальний результат
          audio = np.concatenate(self._speech_buffer)
          audio = preprocess_audio(audio)
          text = self.asr.transcribe(audio)
          self._speech_buffer = []
          return {"text": text, "lang": self.source_lang, "is_final": True}

      return None

  def run(self, callback):
      """Головний цикл — слухає мікрофон і викликає callback з результатом."""
      self._running = True
      self.audio_stream.start()

      try:
          while self._running:
              chunk = self.audio_stream.get_chunk(timeout=1.0)
              if chunk is None:
                  continue
              result = self.process_chunk(chunk)
              if result:
                  callback(result)
      except KeyboardInterrupt:
          print("\nStopping...")
      finally:
          self.stop()

  def set_language(self, lang_code):
      """Змінити мову розпізнавання"""
      lang_config = SUPPORTED_LANGUAGES[lang_code]
      self.asr.set_language(lang_config["mms_code"])
      self.source_lang = lang_code
      print(f"Language changed to: {lang_config['name']}")

  def stop(self):
      """Зупинити pipeline"""
      self._running = False
      self.audio_stream.stop()