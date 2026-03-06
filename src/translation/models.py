"""NLLB-200 модель для перекладу"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config import TRANSLATION_MODEL, DEVICE, USE_FP16


class NLLBTranslator:
  """Переклад через NLLB-200-distilled-600M"""

  def __init__(self, device=DEVICE):
      self.device = device
      print(f"Loading translation model: {TRANSLATION_MODEL}...")
      self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
      self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)

      if USE_FP16 and device == "cuda":
          self.model = self.model.half()

      self.model.to(device)
      self.model.eval()
      print("Translation model loaded")

  def translate(self, text, source_lang, target_lang):
      """Перекласти текст.

      source_lang / target_lang у форматі NLLB: "ukr_Cyrl", "eng_Latn"
      """
      if not text.strip():
          return ""

      self.tokenizer.src_lang = source_lang
      inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

      target_lang_id = self.tokenizer.convert_tokens_to_ids(target_lang)

      with torch.no_grad():
          generated = self.model.generate(
              **inputs,
              forced_bos_token_id=target_lang_id,
              max_new_tokens=256,
          )

      result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
      return result