"""Базова перевірка орфографії"""

import re


class SpellingCorrector:
  """Простий коректор — прибирає повтори та зайві символи"""

  def __init__(self):
      # Типові помилки ASR моделей
      self.replacements = {
          "  ": " ",
          ",,": ",",
          "..": ".",
          "??": "?",
      }

  def correct(self, text):
      """Базова корекція тексту після ASR"""
      if not text.strip():
          return ""

      # Прибираємо повтори пунктуації
      for old, new in self.replacements.items():
          while old in text:
              text = text.replace(old, new)

      # Прибираємо пробіли перед пунктуацією
      text = re.sub(r'\s+([.,?!])', r'\1', text)

      # Прибираємо зайві пробіли
      text = re.sub(r'\s+', ' ', text).strip()

      return text