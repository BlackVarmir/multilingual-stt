"""Обробка абревіатур у тексті"""

import json
import re
from pathlib import Path


class AbbreviationHandler:
  """Знаходить і обробляє абревіатури в тексті"""

  def __init__(self):
      db_path = Path(__file__).parent / "db.json"
      with open(db_path, "r", encoding="utf-8") as f:
          self.db = json.load(f)

  def detect(self, text):
      """Знайти абревіатури в тексті"""
      found = []
      words = re.findall(r'\b[A-Z]{2,}\b', text)
      for word in words:
          if word in self.db:
              found.append(word)
      return found

  def process(self, text, action="keep_original"):
      """Обробити абревіатури в тексті.

      action:
        "keep_original" — залишити як є
        "expand_en" — розгорнути англійською
        "expand_uk" — розгорнути українською
      """
      if action == "keep_original":
          return text

      for abbr, info in self.db.items():
          if abbr in text:
              if action == "expand_en":
                  replacement = f"{abbr} ({info['full']})"
              elif action == "expand_uk":
                  replacement = f"{abbr} ({info['uk']})"
              else:
                  continue
              text = text.replace(abbr, replacement)

      return text

  def add_abbreviation(self, abbr, full, uk):
      """Додати нову абревіатуру"""
      self.db[abbr] = {"full": full, "uk": uk}
      db_path = Path(__file__).parent / "db.json"
      with open(db_path, "w", encoding="utf-8") as f:
          json.dump(self.db, f, ensure_ascii=False, indent=4)