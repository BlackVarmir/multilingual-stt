"""Evaluation Whisper cv6 z beam search + KenLM rescoring na Common Voice test"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import csv
import argparse
import math
import torch
import torchaudio
import kenlm
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from evaluate import load as load_metric


def load_model(base_model_name: str, lora_path: str, device: str):
    """Zavantazhuie Whisper z LoRA adapterom"""
    print(f"Loading {base_model_name}...")
    processor = WhisperProcessor.from_pretrained(base_model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print(f"Loading LoRA from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.to(device).eval()
    return model, processor


def load_test_samples(cv_dir: str, n_samples: int, seed: int = 42):
    """Zavantazhuie zrazky z Common Voice test"""
    csv.field_size_limit(10 * 1024 * 1024)
    rows = []
    tsv_path = f"{cv_dir}/test.tsv"
    clips_dir = f"{cv_dir}/clips/"
    with open(tsv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append({"audio_path": clips_dir + row["path"], "sentence": row["sentence"]})

    import random
    random.seed(seed)
    if n_samples < len(rows):
        rows = random.sample(rows, n_samples)
    return rows


def load_audio(path: str):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.squeeze().numpy()


@torch.no_grad()
def generate_beam(model, processor, audio_array, num_beams: int, num_return_sequences: int, device: str):
    """Generaciya N-best hipotez cherez beam search"""
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    if device == "cuda":
        input_features = input_features.half()

    # Diverse beam search: groups force vidchutno rizni hipotezy
    outputs = model.generate(
        input_features,
        num_beams=num_beams,
        num_beam_groups=num_beams,
        diversity_penalty=1.0,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=225,
        language="uk",
        task="transcribe",
        custom_generate="transformers-community/group-beam-search",
        trust_remote_code=True,
    )

    sequences = outputs.sequences
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        whisper_scores = outputs.sequences_scores.cpu().tolist()
    else:
        whisper_scores = [0.0] * sequences.shape[0]

    texts = processor.batch_decode(sequences, skip_special_tokens=True)
    return texts, whisper_scores


def rescore(hypotheses, whisper_scores, lm_model, alpha: float, beta: float):
    """Pererahovuie najkrashchu hipotezu: whisper_score + alpha * lm_score + beta * len_words"""
    best_text = hypotheses[0]
    best_score = -float("inf")
    for text, w_score in zip(hypotheses, whisper_scores):
        text_clean = text.strip()
        if not text_clean:
            continue
        # KenLM povertaye log10 — konvertuiemo do natural log
        lm_score = lm_model.score(text_clean.lower(), bos=True, eos=True) * math.log(10)
        n_words = len(text_clean.split())
        final = w_score + alpha * lm_score + beta * n_words
        if final > best_score:
            best_score = final
            best_text = text_clean
    return best_text


def evaluate(model, processor, samples, lm_model, num_beams: int, alpha: float, beta: float, device: str):
    """Provodyt evaluation: greedy, beam-only, beam+kenlm"""
    wer_metric = load_metric("wer")
    refs, greedy_preds, beam_preds, rescore_preds = [], [], [], []

    for sample in tqdm(samples, desc="Evaluating"):
        audio = load_audio(sample["audio_path"])
        ref = sample["sentence"].strip().lower()

        # Greedy
        with torch.no_grad():
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            if device == "cuda":
                input_features = input_features.half()
            greedy_ids = model.generate(input_features, max_new_tokens=225,
                                        language="uk", task="transcribe", num_beams=1)
            greedy_text = processor.batch_decode(greedy_ids, skip_special_tokens=True)[0].strip().lower()

        # Beam search N-best
        hyps, w_scores = generate_beam(model, processor, audio,
                                       num_beams=num_beams, num_return_sequences=num_beams,
                                       device=device)
        hyps = [h.strip().lower() for h in hyps]
        beam_text = hyps[0]
        rescore_text = rescore(hyps, w_scores, lm_model, alpha, beta).lower()

        refs.append(ref)
        greedy_preds.append(greedy_text)
        beam_preds.append(beam_text)
        rescore_preds.append(rescore_text)

    return {
        "greedy": wer_metric.compute(predictions=greedy_preds, references=refs),
        "beam": wer_metric.compute(predictions=beam_preds, references=refs),
        "beam_kenlm": wer_metric.compute(predictions=rescore_preds, references=refs),
    }


def tune_alpha_beta(model, processor, samples, lm_model, num_beams: int, device: str):
    """Grid search alpha/beta na nebagatomu pidnabori"""
    print("\nPre-computing beam hypotheses for tuning...")
    cached = []
    wer_metric = load_metric("wer")
    refs = []
    for sample in tqdm(samples, desc="Beam search"):
        audio = load_audio(sample["audio_path"])
        hyps, w_scores = generate_beam(model, processor, audio,
                                       num_beams=num_beams, num_return_sequences=num_beams,
                                       device=device)
        hyps = [h.strip().lower() for h in hyps]
        cached.append((hyps, w_scores))
        refs.append(sample["sentence"].strip().lower())

    # Diagnostyka: skilky unikalnykh hipotez na zrazok
    unique_counts = [len(set(h)) for h, _ in cached]
    avg_unique = sum(unique_counts) / len(unique_counts)
    print(f"\nDiagnostics: avg unique hypotheses per sample = {avg_unique:.2f} / {num_beams}")
    print(f"Samples with all identical hypotheses: {sum(1 for c in unique_counts if c == 1)}/{len(cached)}")
    print(f"\nExample (first 3 samples):")
    for idx in range(min(3, len(cached))):
        hyps, w_scores = cached[idx]
        print(f"\n  Sample {idx}, ref: {refs[idx]!r}")
        for i, (h, s) in enumerate(zip(hyps, w_scores)):
            lm_s = lm_model.score(h, bos=True, eos=True)
            print(f"    [{i}] w_score={s:.4f} lm_log10={lm_s:.2f} text={h!r}")

    best = (1.0, None, None)
    print("\nGrid search alpha/beta:")
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for beta in [0.0, 1.0, 3.0, 5.0]:
            preds = [rescore(h, s, lm_model, alpha, beta).lower() for h, s in cached]
            wer = wer_metric.compute(predictions=preds, references=refs)
            print(f"  alpha={alpha}, beta={beta}: WER={wer:.4f}")
            if wer < best[0]:
                best = (wer, alpha, beta)

    print(f"\nBest: alpha={best[1]}, beta={best[2]}, WER={best[0]:.4f}")
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--lora-path", default="BlackVarmir/whisper-uk-lora-cv6")
    parser.add_argument("--kenlm-path", default="/workspace/multilingual-stt/models/uk_5gram_wiki.bin")
    parser.add_argument("--cv-dir", default="/workspace/multilingual-stt/data/common_voice/cv-corpus-24.0-2025-12-05/uk")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tune", action="store_true", help="Grid search alpha/beta na 100 zrazkakh")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, processor = load_model(args.base_model, args.lora_path, device)
    print(f"Loading KenLM from {args.kenlm_path}...")
    lm_model = kenlm.Model(args.kenlm_path)

    samples = load_test_samples(args.cv_dir, args.num_samples)
    print(f"Loaded {len(samples)} test samples")

    if args.tune:
        tune_samples = samples[:100]
        alpha, beta = tune_alpha_beta(model, processor, tune_samples, lm_model,
                                      args.num_beams, device)
    else:
        alpha, beta = args.alpha, args.beta

    print(f"\nEvaluating with alpha={alpha}, beta={beta}, num_beams={args.num_beams}")
    results = evaluate(model, processor, samples, lm_model,
                       args.num_beams, alpha, beta, device)

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"  Greedy WER:      {results['greedy']:.4f} ({results['greedy']*100:.2f}%)")
    print(f"  Beam search WER: {results['beam']:.4f} ({results['beam']*100:.2f}%)")
    print(f"  Beam + KenLM:    {results['beam_kenlm']:.4f} ({results['beam_kenlm']*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
