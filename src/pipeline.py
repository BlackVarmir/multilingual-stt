"""Головний streaming pipeline для розпізнавання мовлення з перекладом"""

import numpy as np
from src.config import SAMPLE_RATE, SUPPORTED_LANGUAGES
from src.audio.capture import AudioStream
from src.audio.vad import VoiceActivityDetector
from src.audio.preprocessing import preprocess_audio
from src.asr.model import ASRModel
from src.lang_detect.detector import AudioLanguageDetector
from src.translation.translator import Translator
from src.abbreviations.handler import AbbreviationHandler


class StreamingSTTPipeline:
  """Streaming Speech-to-Text Pipeline з перекладом"""

  def __init__(self, source_lang="uk", target_lang="uk", device="cpu",
               auto_detect_lang=False):
      self.source_lang = source_lang
      self.target_lang = target_lang
      self.auto_detect_lang = auto_detect_lang

      print("Initializing pipeline...")
      self.audio_stream = AudioStream()
      self.vad = VoiceActivityDetector()

      lang_config = SUPPORTED_LANGUAGES[source_lang]
      self.asr = ASRModel(lang=lang_config["mms_code"], device=device)

      # LID — тільки якщо автовизначення мови
      self.lid = None
      if auto_detect_lang:
          self.lid = AudioLanguageDetector(device=device)

      # Переклад — тільки якщо мови різні
      self.translator = Translator(device=device)
      self.abbreviations = AbbreviationHandler()

      self._running = False
      self._speech_buffer = []
      print("Pipeline ready!")

  def _detect_language(self, audio):
      """Визначити мову аудіо і оновити ASR"""
      if not self.lid:
          return self.source_lang

      lang_code, confidence = self.lid.detect(audio)

      # Знайти відповідний ключ в SUPPORTED_LANGUAGES
      for key, config in SUPPORTED_LANGUAGES.items():
          if config["mms_code"] == lang_code:
              if key != self.source_lang and confidence > 0.7:
                  self.source_lang = key
                  self.asr.set_language(config["mms_code"])
                  print(f"\nLanguage detected: {config['name']} ({confidence:.0%})")
              return key

      return self.source_lang

  def process_chunk(self, audio_chunk):
      """Обробити один чанк аудіо"""
      is_speaking, just_started, just_ended = self.vad.update(audio_chunk)

      if just_started:
          self._speech_buffer = []

      if is_speaking:
          self._speech_buffer.append(audio_chunk)
          # Partial result — швидко, без перекладу
          audio = np.concatenate(self._speech_buffer)
          audio = preprocess_audio(audio)
          text = self.asr.transcribe(audio)
          return {"text": text, "lang": self.source_lang, "is_final": False}

      if just_ended and self._speech_buffer:
          # Final result — з перекладом
          audio = np.concatenate(self._speech_buffer)
          audio = preprocess_audio(audio)

          if self.auto_detect_lang:
              self._detect_language(audio)

          text = self.asr.transcribe(audio)
          text = self.abbreviations.process(text)

          translated = self.translator.translate(
              text, self.source_lang, self.target_lang
          )

          self._speech_buffer = []
          return {
              "text": translated,
              "original": text,
              "lang": self.source_lang,
              "is_final": True,
          }

      return None

  def run(self, callback):
      """Головний цикл"""
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