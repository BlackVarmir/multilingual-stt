"""Підрахунок WER: greedy vs beam search + KenLM"""

import csv, random, torch, torchaudio
from evaluate import load
from transformers import Wav2Vec2ForCTC, AutoProcessor
from src.decoder.beam_search import BeamSearchDecoder

csv.field_size_limit(10*1024*1024)
wer_metric = load("wer")

model_path = "models/mms-finetuned-cv3"
processor = AutoProcessor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

beam_decoder = BeamSearchDecoder(processor, kenlm_model_path="models/uk_5gram_wiki.bin")
clips_dir = "data/common_voice/cv-corpus-24.0-2025-12-05/uk/clips/"

# Взяти 100 випадкових тестових записів
rows = []
with open("data/common_voice/cv-corpus-24.0-2025-12-05/uk/test.tsv", encoding="utf-8") as f:
  for row in csv.DictReader(f, delimiter="\t"):
      rows.append((row["path"], row["sentence"]))
random.seed(42)
samples = random.sample(rows, 100)

refs, greedy_preds, beam_preds = [], [], []
for i, (fname, ref) in enumerate(samples):
  try:
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

      refs.append(ref.lower())
      greedy_preds.append(greedy_text.lower())
      beam_preds.append(beam_text.lower())

      if (i+1) % 20 == 0:
          print(f"Processed {i+1}/100")
  except Exception as e:
      print(f"Skip {fname}: {e}")

greedy_wer = wer_metric.compute(predictions=greedy_preds, references=refs)
beam_wer = wer_metric.compute(predictions=beam_preds, references=refs)
print(f"\nGreedy WER: {greedy_wer:.4f} ({greedy_wer*100:.1f}%)")
print(f"Beam+KenLM WER: {beam_wer:.4f} ({beam_wer*100:.1f}%)")
print(f"Improvement: {(greedy_wer - beam_wer)*100:.1f}%")
