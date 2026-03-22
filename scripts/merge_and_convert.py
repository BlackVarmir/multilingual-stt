"""Мердж LoRA ваг з базовою моделлю і конвертація в CTranslate2"""

import argparse
from pathlib import Path


def merge_lora(base_model, lora_path, output_path):
    """Об'єднати LoRA адаптер з базовою моделлю"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    processor = WhisperProcessor.from_pretrained(base_model)

    print(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print("Merged model saved!")
    return output_path


def convert_to_ct2(model_path, output_path, quantization="int8"):
    """Конвертація в CTranslate2 для faster-whisper"""
    import subprocess
    print(f"Converting to CTranslate2 ({quantization})...")
    cmd = [
        "ct2-whisper-converter",
        "--model", model_path,
        "--output_dir", output_path,
        "--quantization", quantization,
    ]
    subprocess.run(cmd, check=True)
    print(f"CTranslate2 model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--lora-path", required=True, help="Шлях до LoRA ваг")
    parser.add_argument("--merged-path", default="models/whisper-uk-merged")
    parser.add_argument("--ct2-path", default="models/whisper-uk-ct2")
    parser.add_argument("--quantization", default="int8", choices=["int8", "float16", "float32"])
    args = parser.parse_args()

    merged = merge_lora(args.base_model, args.lora_path, args.merged_path)
    convert_to_ct2(merged, args.ct2_path, args.quantization)
    print("\nAll done! Use with faster-whisper:")
    print(f'  model = WhisperModel("{args.ct2_path}", device="cpu", compute_type="int8")')
