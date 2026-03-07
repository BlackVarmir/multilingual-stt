"""Порівняння швидкості ONNX vs PyTorch"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np

audio = np.random.randn(8000).astype(np.float32) * 0.1

# ONNX
from src.asr.inference_onnx import ONNXASRModel
onnx_model = ONNXASRModel()
for _ in range(3):
  onnx_model.transcribe(audio)
start = time.time()
for _ in range(10):
  onnx_model.transcribe(audio)
onnx_time = (time.time() - start) / 10
print(f"ONNX: {onnx_time*1000:.0f} ms per chunk")

# PyTorch
from src.asr.model import ASRModel
torch_model = ASRModel(lang="ukr", device="cpu")
for _ in range(3):
  torch_model.transcribe(audio)
start = time.time()
for _ in range(10):
  torch_model.transcribe(audio)
torch_time = (time.time() - start) / 10
print(f"PyTorch: {torch_time*1000:.0f} ms per chunk")
print(f"Speedup: {torch_time/onnx_time:.1f}x")