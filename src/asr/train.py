"""Fine-tuning MMS моделі на Common Voice"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_from_disk, Audio
from transformers import (
  Wav2Vec2ForCTC,
  AutoProcessor,
  TrainingArguments,
  Trainer,
)
from src.asr.augmentation import AudioAugmentor


@dataclass
class DataCollatorCTCWithPadding:
  """Collator для CTC тренування — паддить input і labels"""
  processor: AutoProcessor
  padding: Union[bool, str] = True

  def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
      input_features = [{"input_values": f["input_values"]} for f in features]
      label_features = [{"input_ids": f["labels"]} for f in features]

      batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
      labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

      # CTC loss ігнорує -100
      labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
      batch["labels"] = labels

      return batch


def prepare_dataset(batch, processor, augmentor=None):
  """Підготовка одного прикладу для тренування"""
  audio = batch["audio"]
  waveform = np.array(audio["array"], dtype=np.float32)

  # Аугментація (тільки для train)
  if augmentor is not None:
      waveform_tensor = augmentor.augment(waveform, level="medium")
      waveform = waveform_tensor.squeeze().numpy()

  batch["input_values"] = processor(
      waveform, sampling_rate=16000
  ).input_values[0]

  batch["labels"] = processor(
      text=batch["sentence"]
  ).input_ids

  return batch


def compute_wer(pred, processor):
  """Обчислити Word Error Rate"""
  from evaluate import load
  wer_metric = load("wer")

  pred_logits = pred.predictions
  pred_ids = np.argmax(pred_logits, axis=-1)

  # Замінити -100 на pad token
  pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

  pred_str = processor.batch_decode(pred_ids)
  label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

  wer = wer_metric.compute(predictions=pred_str, references=label_str)
  return {"wer": wer}


def main():
  # Параметри
  model_name = "/workspace/multilingual-stt/models/mms-finetuned"
  lang = "ukr"
  output_dir = "/workspace/multilingual-stt/models/mms-finetuned-cv"
  data_dir = "/workspace/multilingual-stt/data/prepared/common_voice_uk"

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device: {device}")

  # Завантаження моделі
  print("Loading model and processor...")
  processor = AutoProcessor.from_pretrained(model_name)
  processor.tokenizer.set_target_lang(lang)

  model = Wav2Vec2ForCTC.from_pretrained(model_name)
  model.load_adapter(lang)

  # Заморозити feature encoder (тренуємо тільки адаптер і CTC head)
  model.freeze_feature_encoder()

  # Завантаження датасету
  print(f"Loading dataset from {data_dir}...")
  dataset = load_from_disk(data_dir)
  train_ds = dataset["train"]
  test_ds = dataset["test"]

  # Ресемплінг до 16kHz
  train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
  test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))

  # Підготовка даних
  augmentor = AudioAugmentor()
  train_ds = train_ds.map(
      lambda b: prepare_dataset(b, processor, augmentor),
      remove_columns=train_ds.column_names,
  )
  test_ds = test_ds.map(
      lambda b: prepare_dataset(b, processor),
      remove_columns=test_ds.column_names,
  )

  # Параметри тренування
  training_args = TrainingArguments(
      output_dir=output_dir,
      per_device_train_batch_size=8,
      gradient_accumulation_steps=2,
      eval_strategy="steps",
      eval_steps=1000,
      save_steps=1000,
      save_total_limit=3,
      num_train_epochs=20,
      learning_rate=1e-5,
      fp16=torch.cuda.is_available(),
      logging_steps=100,
      load_best_model_at_end=True,
      metric_for_best_model="wer",
      greater_is_better=False,
      warmup_steps=500,
      report_to="none",
  )

  # Тренер
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=test_ds,
      data_collator=DataCollatorCTCWithPadding(processor=processor),
      compute_metrics=lambda pred: compute_wer(pred, processor),
      processing_class=processor,
  )

  # Тренування
  print("Starting training...")
  trainer.train()

  # Збереження
  print("Saving model...")
  trainer.save_model(output_dir)
  processor.save_pretrained(output_dir)
  print(f"Model saved to {output_dir}")

  # Фінальна оцінка
  results = trainer.evaluate()
  print(f"\nFinal WER: {results['eval_wer']:.4f}")
  print("Done!")


if __name__ == "__main__":
  main()