"""Квантизація ONNX моделі в INT8"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_mms(input_path="models/mms_ctc.onnx", output_dir="models"):
  """Квантизація MMS ONNX моделі FP32 → INT8"""
  output_dir = Path(output_dir)
  output_path = output_dir / "mms_ctc_int8.onnx"

  input_size = Path(input_path).stat().st_size / (1024 * 1024)
  print(f"Input model: {input_size:.0f} MB")

  print("Quantizing to INT8...")
  quantize_dynamic(
      model_input=str(input_path),
      model_output=str(output_path),
      weight_type=QuantType.QInt8,
  )

  output_size = output_path.stat().st_size / (1024 * 1024)
  print(f"Quantized model: {output_size:.0f} MB")
  print(f"Compression: {input_size / output_size:.1f}x")

  return output_path


if __name__ == "__main__":
  quantize_mms()