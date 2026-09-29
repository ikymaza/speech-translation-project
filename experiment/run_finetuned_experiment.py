"""
Eksperimen tahap FIXED WINDOW dulu (adaptive menyusul di step berikutnya).

Jalur 1 (WER & BLEU): Audio bersih -> ASR pretrained -> teks ID -> MT
pretrained -> teks EN, dibandingkan ke ground truth & referensi Google
Translate. TIDAK lewat NDTW (lihat diskusi kompatibilitas Transformer).

Jalur 2 (kontribusi NDTW): Audio bersih -> Wav2Vec2 -> NDTW fixed window
(beberapa nilai r) -> catat tingkat kegagalan, D_norm, RTF.

== VERSI PATCH ==
Menggunakan model ASR Whisper HASIL FINE-TUNING (checkpoint-2000) alih-alih
Whisper-small dasar, agar BLEU Score konsisten dengan WER 12,79% yang sudah
dilaporkan. Perubahan HANYA pada bagian loading model ASR di run_jalur1();
seluruh bagian lain (MT, normalisasi, perhitungan WER/BLEU, Jalur 2) identik
dengan skrip asli agar perbandingan tetap apple-to-apple.
"""
import sys
import time
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import jiwer

from tqdm import tqdm

from vad.energy_vad import detect_speech_segments, trim_edges_only
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio
from alignment.ndtw import dtw_align

import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# ============ CONFIG ============
QUERIES_DIR = Path("data/queries")
QUERIES_METADATA = Path("data/prepared/queries_metadata.csv")
TRANSLATIONS_PATH = Path("data/translations.csv")
REFERENCE_FEATURES_PATH = Path("data/acoustic_anchor.npy")

# --- PATCH: pakai checkpoint fine-tuned, bukan Whisper-small dasar ---
ASR_MODEL = "model/whisper_finetuned"              # <-- SESUAIKAN path ini kalau lokasi folder checkpoint kamu beda
ASR_BASE_FOR_TOKENIZER = "openai/whisper-small"     # tokenizer diambil dari sini (checkpoint tidak menyimpan tokenizer sendiri)

MT_MODEL = "Helsinki-NLP/opus-mt-id-en"

FIXED_R_CANDIDATES = [10, 20, 30, 50, 70, 100]

# --- PATCH: nama file output BEDA, supaya hasil zero-shot lama (BLEU 34,79) TIDAK tertimpa ---
OUTPUT_JALUR1 = Path("experiment/results_jalur1_wer_bleu_finetuned.csv")
OUTPUT_JALUR2 = Path("experiment/results_jalur2_fixed_window.csv")
# =================================

def run_jalur1(df_meta, df_trans):
    """WER & BLEU dari ASR Whisper (fine-tuned) + MT pretrained, tanpa NDTW."""
    from transformers import pipeline as hf_pipeline
    from transformers import MarianMTModel, MarianTokenizer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
    import torch
    from sacrebleu import sentence_bleu

    device = 0 if torch.cuda.is_available() else -1
    print(f"Memuat model ASR (Whisper fine-tuned checkpoint)... device={'GPU' if device==0 else 'CPU'}")

    # --- PATCH: deteksi tokenizer, fallback ke base model kalau checkpoint tidak menyimpannya ---
    has_tokenizer_files = any(
        os.path.exists(os.path.join(ASR_MODEL, f)) for f in ["tokenizer.json", "vocab.json", "tokenizer_config.json"]
    )
    if has_tokenizer_files:
        print("Tokenizer ditemukan di dalam folder checkpoint.")
        processor = WhisperProcessor.from_pretrained(ASR_MODEL)
    else:
        print(f"Tokenizer tidak ada di checkpoint, mengambil dari base model '{ASR_BASE_FOR_TOKENIZER}'.")
        tokenizer = WhisperTokenizer.from_pretrained(ASR_BASE_FOR_TOKENIZER, language="indonesian", task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(ASR_MODEL)
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL)

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        generate_kwargs={
            "language": "indonesian",
            "task": "transcribe",
            "condition_on_prev_tokens": False,   # cegah Whisper "terjebak" konteks sebelumnya
            "repetition_penalty": 1.3,            # hukum pengulangan kata
            "no_repeat_ngram_size": 3,            # larang 3 kata sama berturut-turut
            "temperature": 0.0,                   # deterministik, hindari drift ke halusinasi
        },
    )

    print("Memuat model MT (Indonesia -> Inggris)...")
    mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
    mt_model = MarianMTModel.from_pretrained(MT_MODEL)
    mt_model.eval()

    merged = df_meta.merge(df_trans, on="file", how="inner")
    results = []

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Jalur 1 (fine-tuned)"):
        wav_path = QUERIES_DIR / row["file"]
        if not wav_path.exists() or pd.isna(row["en_reference"]):
            continue
        try:
            signal, sr = load_audio(str(wav_path))
            clean_signal = trim_edges_only(signal, sr)
            if len(clean_signal) == 0:
                continue

            # ASR pakai Whisper fine-tuned
            asr_output = asr(clean_signal.astype("float32"))
            predicted_id_text = asr_output["text"].strip()

            # MT (sama seperti sebelumnya)
            mt_inputs = mt_tokenizer(predicted_id_text, return_tensors="pt", padding=True)
            translated = mt_model.generate(**mt_inputs)
            predicted_en_text = mt_tokenizer.decode(translated[0], skip_special_tokens=True)

            wer_score = jiwer.wer(normalize_text(str(row["sentence_id"])), normalize_text(predicted_id_text))
            bleu_score = sentence_bleu(predicted_en_text, [str(row["en_reference"])]).score

            results.append({
                "file": row["file"],
                "ground_truth_id": row["sentence_id"],
                "predicted_id": predicted_id_text,
                "reference_en": row["en_reference"],
                "predicted_en": predicted_en_text,
                "wer": wer_score,
                "bleu": bleu_score,
            })
        except Exception as e:
            print(f"Gagal proses {row['file']}: {e}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_JALUR1, index=False)
    print(f"\nJalur 1 (fine-tuned) selesai. {len(out_df)} sampel berhasil diproses.")
    if len(out_df) > 0:
        print(f"Rata-rata WER : {out_df['wer'].mean():.4f}  (bandingkan ke 0.1279 / 12,79%)")
        print(f"Rata-rata BLEU: {out_df['bleu'].mean():.2f}  (bandingkan ke 34.79 versi zero-shot)")
    return out_df


def main():
    df_meta = pd.read_csv(QUERIES_METADATA)
    df_trans = pd.read_csv(TRANSLATIONS_PATH)

    run_jalur1(df_meta, df_trans)
    # --- PATCH: Jalur 2 (NDTW fixed window) TIDAK diulang, sudah pernah dijalankan sebelumnya ---
    # run_jalur2(df_meta)


if __name__ == "__main__":
    main()
