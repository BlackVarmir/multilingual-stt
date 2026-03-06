"""CLI для Multilingual Speech-to-Text"""

import argparse
from src.config import SUPPORTED_LANGUAGES, DEVICE


def print_result(result):
  """Callback для виводу результатів"""
  if result["is_final"]:
      original = result.get("original", "")
      if original and original != result["text"]:
          print(f"\n[{result['lang']}] {original}")
          print(f">>> {result['text']}")
      else:
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
      "--target-lang", default=None,
      choices=SUPPORTED_LANGUAGES.keys(),
      help="Мова перекладу (default: same as source)"
  )
  parser.add_argument(
      "--auto-detect", action="store_true",
      help="Автоматичне визначення мови"
  )
  parser.add_argument(
      "--device", default=DEVICE,
      choices=["cpu", "cuda"],
      help=f"Пристрій для inference (default: {DEVICE})"
  )
  args = parser.parse_args()

  target = args.target_lang or args.source_lang
  print(f"Source: {SUPPORTED_LANGUAGES[args.source_lang]['name']}")
  print(f"Target: {SUPPORTED_LANGUAGES[target]['name']}")
  print(f"Auto-detect: {args.auto_detect}")
  print(f"Device: {args.device}")
  print("Press Ctrl+C to stop\n")

  from src.pipeline import StreamingSTTPipeline
  pipeline = StreamingSTTPipeline(
      source_lang=args.source_lang,
      target_lang=target,
      device=args.device,
      auto_detect_lang=args.auto_detect,
  )
  pipeline.run(callback=print_result)


if __name__ == "__main__":
  main()