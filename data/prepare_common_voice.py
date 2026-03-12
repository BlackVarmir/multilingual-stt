"""Підготовка Common Voice датасету для тренування"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
csv.field_size_limit(10 * 1024 * 1024)  # 10MB ліміт
import torch
import torchaudio
from datasets import Dataset, Audio, DatasetDict

CV_DIR = Path("data/common_voice/cv-corpus-24.0-2025-12-05/uk")
OUTPUT_DIR = Path("data/prepared/common_voice_uk")


def load_tsv(tsv_path):
  """Завантажити TSV файл Common Voice"""
  rows = []
  with open(tsv_path, "r", encoding="utf-8") as f:
      reader = csv.DictReader(f, delimiter="\t")
      for row in reader:
          clip_path = CV_DIR / "clips" / row["path"]
          if clip_path.exists():
              rows.append({
                  "audio": str(clip_path),
                  "sentence": row["sentence"],
              })
  return rows


def main():
  print("Loading Common Voice TSV files...")
  train_data = load_tsv(CV_DIR / "train.tsv")
  test_data = load_tsv(CV_DIR / "test.tsv")
  dev_data = load_tsv(CV_DIR / "dev.tsv")

  print(f"Train: {len(train_data)}, Test: {len(test_data)}, Dev: {len(dev_data)}")

  print("Creating HuggingFace datasets...")
  train_ds = Dataset.from_list(train_data).cast_column("audio", Audio(sampling_rate=16000))
  test_ds = Dataset.from_list(test_data).cast_column("audio", Audio(sampling_rate=16000))
  dev_ds = Dataset.from_list(dev_data).cast_column("audio", Audio(sampling_rate=16000))

  ds = DatasetDict({
      "train": train_ds,
      "test": test_ds,
      "validation": dev_ds,
  })

  print(f"Saving to {OUTPUT_DIR}...")
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  ds.save_to_disk(str(OUTPUT_DIR))
  print("Done!")


if __name__ == "__main__":
  main()