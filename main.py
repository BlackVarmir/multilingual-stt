"""CLI для Multilingual Speech-to-Text"""

import argparse
from src.config import SUPPORTED_LANGUAGES, DEVICE


def print_result(result):
  """Callback для виводу результатів"""
  if result["is_final"]:
      print(f"\n>>> {result['text']}")
  else:
      print(f"\r... {result['text']}", end="", flush=True)


def main():
  parser = argparse.ArgumentParser(description="Multilingual Speech-to-Text")
  parser.add_argument(
      "--source-lang", default="uk",
      choices=SUPPORTED_LANGUAGES.keys(),
      help="Мова розпізнавання (default: uk)"
  )
  parser.add_argument(
      "--device", default=DEVICE,
      choices=["cpu", "cuda"],
      help=f"Пристрій для inference (default: {DEVICE})"
  )
  args = parser.parse_args()

  print(f"Source language: {SUPPORTED_LANGUAGES[args.source_lang]['name']}")
  print(f"Device: {args.device}")
  print("Press Ctrl+C to stop\n")

  from src.pipeline import StreamingSTTPipeline
  pipeline = StreamingSTTPipeline(
      source_lang=args.source_lang,
      device=args.device,
  )
  pipeline.run(callback=print_result)


if __name__ == "__main__":
  main()