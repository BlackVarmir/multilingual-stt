"""Підбір alpha/beta для beam search + KenLM"""

import csv, random, torch, torchaudio
from evaluate import load
from transformers import Wav2Vec2ForCTC, AutoProcessor
from src.decoder.beam_search import BeamSearchDecoder

csv.field_size_limit(10*1024*1024)
wer_metric = load("wer")

model_path = "models/mms-finetuned-cv2"
processor = AutoProcessor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

clips_dir = "data/common_voice/cv-corpus-24.0-2025-12-05/uk/clips/"
rows = []
with open("data/common_voice/cv-corpus-24.0-2025-12-05/uk/dev.tsv", encoding="utf-8") as f:
  for row in csv.DictReader(f, delimiter="\t"):
      rows.append((row["path"], row["sentence"]))
random.seed(42)
samples = random.sample(rows, 50)

# Попередньо обчислити logits
print("Computing logits...")
data = []
for fname, ref in samples:
  try:
      waveform, sr = torchaudio.load(clips_dir + fname)
      if sr != 16000:
          waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
      waveform = waveform.squeeze().numpy()
      inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
      with torch.no_grad():
          logits = model(inputs.input_values).logits
      data.append((logits, ref.lower()))
  except:
      pass
print(f"Got {len(data)} samples")

best_wer, best_a, best_b = 1.0, 0, 0
for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
  for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
      decoder = BeamSearchDecoder(processor, "models/uk_5gram.bin", alpha=alpha, beta=beta)
      preds = []
      refs = []
      for logits, ref in data:
          text = decoder.decode(logits)[0].lower()
          preds.append(text)
          refs.append(ref)
      wer = wer_metric.compute(predictions=preds, references=refs)
      if wer < best_wer:
          best_wer = wer
          best_a, best_b = alpha, beta
      print(f"alpha={alpha}, beta={beta}: WER={wer:.4f}")

print(f"\nBest: alpha={best_a}, beta={best_b}, WER={best_wer:.4f} ({best_wer*100:.1f}%)")
