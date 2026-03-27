"""Fine-tuning Whisper з LoRA на Common Voice Ukrainian"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import csv
import torch
import torchaudio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, concatenate_datasets
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
    """Collator: zavantazhuie audio, rakhuje mel, paddit labels"""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            waveform, sr = torchaudio.load(f["audio_path"])
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            mel = self.processor.feature_extractor(
                waveform.squeeze().numpy(), sampling_rate=16000
            ).input_features[0]
            input_features.append({"input_features": mel})

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """Tokenize text only, audio loaded in collator"""
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def main():
    model_name = "openai/whisper-large-v3-turbo"
    output_dir = "/workspace/multilingual-stt/models/whisper-uk-lora"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Gradient checkpointing — зменшує VRAM в 2-3x
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="uk", task="transcribe"
    )
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load Common Voice directly from TSV
    cv_dir = "/workspace/multilingual-stt/data/common_voice/cv-corpus-24.0-2025-12-05/uk"
    csv.field_size_limit(10 * 1024 * 1024)

    def load_cv_tsv(split_name):
        rows = []
        tsv_path = f"{cv_dir}/{split_name}.tsv"
        clips_dir = f"{cv_dir}/clips/"
        with open(tsv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows.append({"audio_path": clips_dir + row["path"], "sentence": row["sentence"]})
        return Dataset.from_list(rows)

    print("Loading Common Voice directly from TSV...")
    train_ds = load_cv_tsv("train")
    dev_ds = load_cv_tsv("dev")
    test_ds = load_cv_tsv("test")
    train_ds = concatenate_datasets([train_ds, dev_ds])
    print(f"Train + dev: {len(train_ds)}, Test: {len(test_ds)}")

    if len(test_ds) > 1000:
        test_ds = test_ds.shuffle(seed=42).select(range(1000))
        print(f"Using 1000 random test samples for eval")

    # Map only tokenizes text (instant, no audio processing)
    print("Tokenizing text...")
    train_ds = train_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=["sentence"],
    )
    test_ds = test_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=["sentence"],
    )

    # Whisper max label length = 448 tokens
    train_before = len(train_ds)
    train_ds = train_ds.filter(lambda x: len(x["labels"]) <= 448)
    test_ds = test_ds.filter(lambda x: len(x["labels"]) <= 448)
    print(f"Filtered: {train_before} -> {len(train_ds)} train, {len(test_ds)} test (max 448 tokens)")

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
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
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

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

    print("Starting LoRA training...")
    trainer.train()

    print("Saving LoRA weights...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    results = trainer.evaluate()
    print(f"\nFinal WER: {results['eval_wer']:.4f}")
    print(f"LoRA weights saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
