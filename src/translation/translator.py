"""Фасад для перекладу"""

from src.config import SUPPORTED_LANGUAGES
from src.translation.models import NLLBTranslator


class Translator:
  """Головний клас перекладу"""

  def __init__(self, device="cpu"):
      self._translator = None
      self._device = device

  def _ensure_loaded(self):
      """Завантажити модель при першому використанні"""
      if self._translator is None:
          self._translator = NLLBTranslator(device=self._device)

  def translate(self, text, source_lang, target_lang):
      """Перекласти текст між мовами.

      source_lang / target_lang — код мови ("uk", "en", "ru")
      Якщо мови однакові — повертає без перекладу.
      """
      if source_lang == target_lang:
          return text

      if not text.strip():
          return ""

      self._ensure_loaded()

      src_nllb = SUPPORTED_LANGUAGES[source_lang]["nllb_code"]
      tgt_nllb = SUPPORTED_LANGUAGES[target_lang]["nllb_code"]

      return self._translator.translate(text, src_nllb, tgt_nllb)