"""Тест: порівняння greedy vs beam search + KenLM на 5 файлах"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, AutoProcessor
from src.decoder.beam_search import BeamSearchDecoder

model_path = "models/mms-finetuned-cv"
processor = AutoProcessor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

beam_decoder = BeamSearchDecoder(processor,
kenlm_model_path="models/uk_5gram.bin")

clips_dir = "data/common_voice/cv-corpus-24.0-2025-12-05/uk/clips/"
samples = [
  ("common_voice_uk_37996449.mp3", "За його прикладом пішли інші."),
  ("common_voice_uk_36947434.mp3", "Чорний дим вирвався з-під стріхи вкупі з полум'ям."),
  ("common_voice_uk_27597891.mp3", "Характерний для гранітів, гранітних пегматитів, розповсюджений мінерал сланців та гнейсів."),
  ("common_voice_uk_34988150.mp3", "Я кажу тобі ще раз."),
  ("common_voice_uk_24030052.mp3", "— Умикну, ось видіти-ймеш."),
]

for fname, reference in samples:
  waveform, sr = torchaudio.load(clips_dir + fname)
  if sr != 16000:
      waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
  waveform = waveform.squeeze().numpy()

  inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
  with torch.no_grad():
      logits = model(inputs.input_values).logits

  greedy_ids = torch.argmax(logits, dim=-1)
  greedy_text = processor.batch_decode(greedy_ids)[0]
  beam_text = beam_decoder.decode(logits)[0]

  print(f"REF:    {reference}")
  print(f"GREEDY: {greedy_text}")
  print(f"BEAM:   {beam_text}")
  print("-" * 60)