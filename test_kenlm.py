"""Тест: порівняння greedy vs beam search + KenLM"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, AutoProcessor
from src.decoder.beam_search import BeamSearchDecoder

# Завантаження моделі
model_path = "models/mms-finetuned-cv"
processor = AutoProcessor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

# Завантаження аудіо
audio_path = "data/common_voice/cv-corpus-24.0-2025-12-05/uk/clips/common_voice_uk_20894281.mp3"
waveform, sr = torchaudio.load(audio_path)
if sr != 16000:
  waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
waveform = waveform.squeeze().numpy()

# Отримати logits
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
  logits = model(inputs.input_values).logits

# Greedy декодер
greedy_ids = torch.argmax(logits, dim=-1)
greedy_text = processor.batch_decode(greedy_ids)[0]

# Beam search + KenLM
beam_decoder = BeamSearchDecoder(processor,
kenlm_model_path="models/uk_5gram.bin")
beam_text = beam_decoder.decode(logits)[0]

print(f"Greedy:      {greedy_text}")
print(f"Beam+KenLM:  {beam_text}")