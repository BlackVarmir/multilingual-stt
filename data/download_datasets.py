"""Завантаження датасетів для fine-tuning"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

SAVE_DIR = Path("data/raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def download_common_voice_uk():
  """Mozilla Common Voice — українська"""
  print("Downloading Common Voice Ukrainian...")
  ds = load_dataset(
      "mozilla-foundation/common_voice_17_0",
      "uk",
      trust_remote_code=True,
  )
  ds.save_to_disk(str(SAVE_DIR / "common_voice_uk"))
  print(f"Ukrainian: {len(ds['train'])} train, {len(ds['test'])} test")
  return ds


def download_common_voice_ru():
  """Mozilla Common Voice — російська (частина)"""
  print("Downloading Common Voice Russian (subset)...")
  ds = load_dataset(
      "mozilla-foundation/common_voice_17_0",
      "ru",
      trust_remote_code=True,
  )
  ds.save_to_disk(str(SAVE_DIR / "common_voice_ru"))
  print(f"Russian: {len(ds['train'])} train, {len(ds['test'])} test")
  return ds


def download_common_voice_en():
  """Mozilla Common Voice — англійська (частина)"""
  print("Downloading Common Voice English (subset)...")
  ds = load_dataset(
      "mozilla-foundation/common_voice_17_0",
      "en",
      trust_remote_code=True,
  )
  ds.save_to_disk(str(SAVE_DIR / "common_voice_en"))
  print(f"English: {len(ds['train'])} train, {len(ds['test'])} test")
  return ds


if __name__ == "__main__":
  print("=== Downloading datasets ===")
  print("NOTE: Common Voice requires HuggingFace authentication.")
  print("Run: huggingface-cli login")
  print()

  try:
      download_common_voice_uk()
  except Exception as e:
      print(f"Error downloading UK: {e}")

  try:
      download_common_voice_ru()
  except Exception as e:
      print(f"Error downloading RU: {e}")

  try:
      download_common_voice_en()
  except Exception as e:
      print(f"Error downloading EN: {e}")

  print("\n=== Done ===")