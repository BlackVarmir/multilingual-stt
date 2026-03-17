"""Beam Search CTC декодер з KenLM мовною моделлю"""

import torch
import numpy as np
from pyctcdecode import build_ctcdecoder


class BeamSearchDecoder:
  """Beam search декодер з KenLM — покращує точність розпізнавання"""

  def __init__(self, processor, kenlm_model_path=None, beam_width=100, alpha=0.5, beta=1.0):
      # Отримати словник з процесора
      vocab = processor.tokenizer.get_vocab()
      sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
      labels = [k for k, v in sorted_vocab]

      # Завантажити unigrams
      unigrams = None
      if kenlm_model_path:
          # Спробувати окремий файл unigrams
          import os
          unigrams_path = os.path.join(os.path.dirname(kenlm_model_path), 'uk_unigrams.txt')
          if os.path.exists(unigrams_path):
              with open(unigrams_path, 'r', encoding='utf-8') as f:
                  unigrams = [line.strip() for line in f if line.strip()]
              print(f"Loaded {len(unigrams)} unigrams from {unigrams_path}")
          elif kenlm_model_path.endswith('.arpa'):
              # Парсити з ARPA
              unigrams = []
              with open(kenlm_model_path, 'r', encoding='utf-8') as f:
                  in_unigrams = False
                  for line in f:
                      if line.startswith('\\1-grams:'):
                          in_unigrams = True
                      elif line.startswith('\\2-grams:') or line.startswith('\\end\\'):
                          break
                      elif in_unigrams and line.strip():
                          parts = line.strip().split('\t')
                          if len(parts) >= 2:
                              unigrams.append(parts[1])
              print(f"Loaded {len(unigrams)} unigrams from ARPA")

      # Побудувати декодер
      self.decoder = build_ctcdecoder(
          labels=labels,
          kenlm_model_path=kenlm_model_path,
          unigrams=unigrams,
          alpha=alpha,
          beta=beta,
      )
      self.beam_width = beam_width

  def decode(self, logits, processor=None):
      """Декодує logits в текст через beam search + KenLM"""
      if isinstance(logits, torch.Tensor):
          log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
          log_probs = log_probs.cpu().numpy()
      else:
          log_probs = logits

      results = []
      for i in range(log_probs.shape[0]):
          text = self.decoder.decode(log_probs[i], beam_width=self.beam_width)
          # Прибрати артефакти та capitalize
          text = text.replace('⁇', '').strip()
          if text:
              text = text[0].upper() + text[1:]
          results.append(text)

      return results
