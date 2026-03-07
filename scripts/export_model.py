"""Експорт MMS моделі в ONNX формат"""

import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, AutoProcessor


def export_mms_to_onnx(lang="ukr", output_dir="models"):
  """Експортує MMS CTC модель в ONNX"""
  output_dir = Path(output_dir)
  output_dir.mkdir(exist_ok=True)
  output_path = output_dir / "mms_ctc.onnx"

  print("Loading MMS model...")
  model_name = "facebook/mms-1b-all"
  processor = AutoProcessor.from_pretrained(model_name)
  model = Wav2Vec2ForCTC.from_pretrained(model_name)

  processor.tokenizer.set_target_lang(lang)
  model.load_adapter(lang)
  model.eval()

  # Зберігаємо processor окремо (потрібен для декодування)
  processor.save_pretrained(str(output_dir / "mms_processor"))
  print("Processor saved")

  # Dummy input — 1 секунда аудіо
  dummy_input = torch.randn(1, 16000)

  print(f"Exporting to ONNX: {output_path}")
  torch.onnx.export(
      model,
      dummy_input,
      str(output_path),
      input_names=["input_values"],
      output_names=["logits"],
      dynamic_axes={
          "input_values": {0: "batch", 1: "time"},
          "logits": {0: "batch", 1: "time"},
      },
      opset_version=17,
  )

  # Перевірка розміру
  size_mb = output_path.stat().st_size / (1024 * 1024)
  print(f"ONNX model saved: {size_mb:.0f} MB")

  # Валідація — порівняти output
  print("Validating ONNX model...")
  import onnxruntime as ort

  session = ort.InferenceSession(str(output_path))
  onnx_input = {"input_values": dummy_input.numpy()}
  onnx_output = session.run(None, onnx_input)[0]

  with torch.no_grad():
      torch_output = model(dummy_input).logits.numpy()

  diff = np.abs(onnx_output - torch_output).max()
  print(f"Max difference PyTorch vs ONNX: {diff:.6f}")
  if diff < 1e-3:
      print("Validation PASSED")
  else:
      print(f"WARNING: difference is large ({diff:.6f})")

  return output_path


if __name__ == "__main__":
  export_mms_to_onnx()