"""Greedy CTC декодер"""

import torch


class GreedyCTCDecoder:
  """Простий greedy декодер — бере argmax на кожному кроці"""

  def decode(self, logits, processor):
      """Декодує logits в текст.

      logits: torch.Tensor [batch, time, vocab]
      processor: Wav2Vec2Processor з токенізатором
      Повертає: список рядків
      """
      predicted_ids = torch.argmax(logits, dim=-1)
      texts = processor.batch_decode(predicted_ids)
      return [t.strip() for t in texts]