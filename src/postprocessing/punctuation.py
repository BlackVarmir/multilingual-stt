"""Автоматична пунктуація тексту"""

from transformers import pipeline


class PunctuationRestorer:
  """Додає розділові знаки та великі літери до тексту"""

  def __init__(self):
      print("Loading punctuation model...")
      self.pipe = pipeline(
          "ner",
          model="oliverguhr/fullstop-punctuation-multilang-large",
          aggregation_strategy="none",
      )
      print("Punctuation model loaded")

  def restore(self, text):
      """Додати пунктуацію до тексту."""
      if not text.strip():
          return ""

      result = self.pipe(text)

      # Збираємо слова з subword токенів
      words = []
      labels = []
      for token in result:
          word = token["word"]
          label = token["entity"]

          if word.startswith("▁") or not words:
              # Новий токен
              words.append(word.strip("▁ "))
              labels.append(label)
          else:
              # Продовження попереднього слова
              words[-1] += word
              labels[-1] = label  # пунктуація від останнього токена

      # Додаємо пунктуацію
      output = []
      for word, label in zip(words, labels):
          if not word:
              continue
          if label in (".", ",", "?"):
              output.append(word + label)
          elif label == "-":
              output.append(word + " -")
          else:
              output.append(word)

      text = " ".join(output)

      # Велика літера на початку і після . ? !
      if text:
          text = text[0].upper() + text[1:]
          for ch in ".?!":
              parts = text.split(f"{ch} ")
              text = f"{ch} ".join(
                  p[0].upper() + p[1:] if p else p for p in parts
              )

      return text