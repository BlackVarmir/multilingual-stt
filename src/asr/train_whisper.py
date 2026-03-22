"""Fine-tuning Whisper з LoRA на Common Voice Ukrainian"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_from_disk, Audio, concatenate_datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator для Whisper — паддить mel features і labels"""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Паддінг input features (mel spectrograms)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Паддінг labels (токенізований текст)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # -100 для паддінгу — loss їх ігнорує
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Прибрати BOS токен якщо модель додає його сама
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """Підготовка одного прикладу для Whisper"""
    audio = batch["audio"]

    # Mel spectrogram (Whisper сам рахує 80-channel log-mel)
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]

    # Токенізація тексту (seq2seq, не CTC)
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

    return batch


def main():
    # Параметри
    model_name = "openai/whisper-large-v3-turbo"
    output_dir = "/workspace/multilingual-stt/models/whisper-uk-lora"
    data_dir = "/workspace/multilingual-stt/data/prepared/common_voice_uk"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Завантаження моделі
    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Налаштування для українській мови
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="uk", task="transcribe"
    )
    model.config.suppress_tokens = []

    # LoRA — тренуємо тільки маленький адаптер, базова модель не змінюється
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Завантаження датасету
    print(f"Loading dataset from {data_dir}...")
    dataset = load_from_disk(data_dir)

    train_ds = dataset["train"]
    if "validation" in dataset:
        train_ds = concatenate_datasets([train_ds, dataset["validation"]])
        print(f"Combined train + validation: {len(train_ds)} samples")
    test_ds = dataset["test"]

    # Менший eval set для швидкості (generate на 10k = години)
    if len(test_ds) > 1000:
        test_ds = test_ds.shuffle(seed=42).select(range(1000))
        print(f"Using 1000 random test samples for eval")

    # Ресемплінг до 16kHz
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Підготовка даних (num_proc=8 для швидкості)
    print("Preparing datasets...")
    train_ds = train_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=train_ds.column_names,
        num_proc=8,
    )
    test_ds = test_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=test_ds.column_names,
        num_proc=8,
    )

    # WER метрика
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Параметри тренування
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        num_train_epochs=5,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        fp16=device == "cuda",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        warmup_ratio=0.1,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=225,
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Тренер
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Тренування
    print("Starting LoRA training...")
    trainer.train()

    # Збереження LoRA ваг
    print("Saving LoRA weights...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Фінальна оцінка
    results = trainer.evaluate()
    print(f"\nFinal WER: {results['eval_wer']:.4f}")
    print(f"LoRA weights saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
