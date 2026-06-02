"""Fine-tuning Whisper з LoRA (cv6): CV + FLEURS + SpecAugment + LoRA rank 64"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import csv
import random
import torch
import torchaudio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import evaluate
import soundfile as sf


def apply_spec_augment(mel: np.ndarray, freq_mask_param: int = 27,
                       time_mask_param: int = 100, num_freq_masks: int = 2,
                       num_time_masks: int = 2) -> np.ndarray:
    """SpecAugment: maskuie chastotni ta chasovi diapazony spectrogramy"""
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0:f0 + f, :] = 0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, n_frames))
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, t0:t0 + t] = 0

    return mel


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator: zavantazhuie audio, rakhuie mel + SpecAugment (opcionalno)"""
    processor: WhisperProcessor
    decoder_start_token_id: int
    apply_augment: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            waveform, sr = torchaudio.load(f["audio_path"])
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            mel = self.processor.feature_extractor(
                waveform.squeeze().numpy(), sampling_rate=16000
            ).input_features[0]

            if self.apply_augment:
                mel = apply_spec_augment(mel)

            input_features.append({"input_features": mel})

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].half()

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class CustomTrainer(Seq2SeqTrainer):
    """Trainer z riznymy collator'amy dlya train (z augment) ta eval (bez)"""
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is not None:
            original = self.data_collator
            self.data_collator = self.eval_data_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = original
        return super().get_eval_dataloader(eval_dataset)


def prepare_dataset(batch, processor):
    """Tilky tokenizuie tekst, audio zavantazhuietsa v collator'i"""
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def load_cv(cv_dir, split_name):
    csv.field_size_limit(10 * 1024 * 1024)
    rows = []
    tsv_path = f"{cv_dir}/{split_name}.tsv"
    clips_dir = f"{cv_dir}/clips/"
    with open(tsv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append({"audio_path": clips_dir + row["path"], "sentence": row["sentence"]})
    return Dataset.from_list(rows)


def prepare_fleurs(out_dir):
    """Zavantazhuie FLEURS uk, zberihaie audio v wav fajly, povertaie Dataset"""
    os.makedirs(out_dir, exist_ok=True)
    print("Loading FLEURS uk_ua...")
    fleurs = load_dataset("google/fleurs", "uk_ua")

    all_rows = []
    for split in ["train", "validation"]:
        for i, sample in enumerate(fleurs[split]):
            audio_path = f"{out_dir}/{split}_{i}.wav"
            if not os.path.exists(audio_path):
                sf.write(audio_path, sample["audio"]["array"],
                         sample["audio"]["sampling_rate"])
            all_rows.append({"audio_path": audio_path,
                             "sentence": sample["transcription"]})

    print(f"FLEURS: {len(all_rows)} samples")
    return Dataset.from_list(all_rows)


def main():
    model_name = "openai/whisper-large-v3-turbo"
    output_dir = "/workspace/multilingual-stt/models/whisper-uk-lora-cv6"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Gradient checkpointing — zmenshuie VRAM v 2-3x
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="uk", task="transcribe"
    )
    model.config.suppress_tokens = []

    # LoRA z bilshym rank (64 zamist 32) — bilshe yemnosti
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    cv_dir = "/workspace/multilingual-stt/data/common_voice/cv-corpus-24.0-2025-12-05/uk"
    fleurs_dir = "/workspace/multilingual-stt/data/fleurs/uk"

    print("Loading Common Voice...")
    cv_train = load_cv(cv_dir, "train")
    cv_dev = load_cv(cv_dir, "dev")
    cv_test = load_cv(cv_dir, "test")

    print("Loading FLEURS...")
    fleurs_train = prepare_fleurs(fleurs_dir)

    train_ds = concatenate_datasets([cv_train, cv_dev, fleurs_train])
    test_ds = cv_test
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    if len(test_ds) > 1000:
        test_ds = test_ds.shuffle(seed=42).select(range(1000))
        print(f"Using 1000 random test samples for eval")

    print("Tokenizing text...")
    train_ds = train_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=["sentence"],
    )
    test_ds = test_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=["sentence"],
    )

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
        num_train_epochs=10,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        fp16=device == "cuda",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        warmup_ratio=0.05,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    train_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_augment=True,
    )
    eval_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_augment=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=train_collator,
        eval_data_collator=eval_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting LoRA training (cv6)...")
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
