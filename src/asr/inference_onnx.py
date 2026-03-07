"""ONNX Runtime inference для MMS ASR"""

import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor
from src.config import SAMPLE_RATE


class ONNXASRModel:
  """MMS ASR через ONNX Runtime (CPU або GPU)"""

  def __init__(self, model_path="models/mms_ctc.onnx",
               processor_path="models/mms_processor", device="cpu"):
      self.device = device
      print(f"Loading ONNX model: {model_path}")

      providers = ["CUDAExecutionProvider"] if device == "cuda" \
                  else ["CPUExecutionProvider"]
      self.session = ort.InferenceSession(model_path, providers=providers)
      self.processor = AutoProcessor.from_pretrained(processor_path)
      print(f"ONNX model loaded ({device})")

  def transcribe(self, audio):
      """Розпізнати мовлення з аудіо.

      audio: numpy array, float32, 16kHz
      Повертає: розпізнаний текст
      """
      inputs = self.processor(
          audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
      )
      input_values = inputs.input_values

      logits = self.session.run(None, {"input_values": input_values})[0]
      predicted_ids = np.argmax(logits, axis=-1)
      text = self.processor.batch_decode(predicted_ids)[0]

      return text.strip()