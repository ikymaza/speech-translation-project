"""
Eksperimen tahap FIXED WINDOW dulu (adaptive menyusul di step berikutnya).

Jalur 1 (WER & BLEU): Audio bersih -> ASR pretrained -> teks ID -> MT
pretrained -> teks EN, dibandingkan ke ground truth & referensi Google
Translate. TIDAK lewat NDTW (lihat diskusi kompatibilitas Transformer).

Jalur 2 (kontribusi NDTW): Audio bersih -> Wav2Vec2 -> NDTW fixed window
(beberapa nilai r) -> catat tingkat kegagalan, D_norm, RTF.
"""
import sys
import time
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

ASR_MODEL = "openai/whisper-small"
MT_MODEL = "Helsinki-NLP/opus-mt-id-en"

FIXED_R_CANDIDATES = [10, 20, 30, 50, 70, 100]

OUTPUT_JALUR1 = Path("experiment/results_jalur1_wer_bleu.csv")
OUTPUT_JALUR2 = Path("experiment/results_jalur2_fixed_window.csv")
# =================================

def run_jalur1(df_meta, df_trans):
    """WER & BLEU dari ASR Whisper + MT pretrained, tanpa NDTW."""
    from transformers import pipeline as hf_pipeline
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    from sacrebleu import sentence_bleu

    device = 0 if torch.cuda.is_available() else -1
    print(f"Memuat model ASR (Whisper)... device={'GPU' if device==0 else 'CPU'}")
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
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

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Jalur 1"):
        wav_path = QUERIES_DIR / row["file"]
        if not wav_path.exists() or pd.isna(row["en_reference"]):
            continue
        try:
            signal, sr = load_audio(str(wav_path))
            clean_signal = trim_edges_only(signal, sr)
            if len(clean_signal) == 0:
                continue

            # ASR pakai Whisper
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
    print(f"\nJalur 1 selesai. {len(out_df)} sampel berhasil diproses.")
    if len(out_df) > 0:
        print(f"Rata-rata WER: {out_df['wer'].mean():.4f}")
        print(f"Rata-rata BLEU: {out_df['bleu'].mean():.2f}")
    return out_df

def run_jalur2(df_meta):
    """Bandingkan beberapa nilai fixed window r: tingkat kegagalan, D_norm, RTF."""
    print("\nMemuat Wav2Vec2 untuk Jalur 2...")
    extractor = Wav2Vec2FeatureExtractor()
    ref_features = np.load(REFERENCE_FEATURES_PATH)

    results = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Jalur 2"):
        wav_path = QUERIES_DIR / row["file"]
        if not wav_path.exists():
            continue
        signal, sr = load_audio(str(wav_path))
        clean_signal = trim_edges_only(signal, sr)
        if len(clean_signal) / sr < 0.5:  # audio bersih < 0.5 detik, terlalu pendek untuk Whisper
            continue

        t0 = time.perf_counter()
        query_features = extractor.extract(clean_signal, sr)
        t_extract = time.perf_counter() - t0
        n, m = query_features.shape[0], ref_features.shape[0]

        for r in FIXED_R_CANDIDATES:
            t0 = time.perf_counter()
            try:
                result = dtw_align(query_features, ref_features, window_size=r)
                t_align = time.perf_counter() - t0
                results.append({
                    "file": row["file"], "N": n, "M": m, "r": r,
                    "status": "ok",
                    "normalized_distance": result.normalized_distance,
                    "rtf_partial": (t_extract + t_align) / row["duration_sec"],
                })
            except ValueError:
                results.append({
                    "file": row["file"], "N": n, "M": m, "r": r,
                    "status": "FAILED", "normalized_distance": None, "rtf_partial": None,
                })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_JALUR2, index=False)

    print(f"\nJalur 2 selesai. Ringkasan tingkat kegagalan per nilai r:")
    summary = out_df.groupby("r")["status"].apply(lambda s: (s == "FAILED").mean() * 100)
    print(summary.round(1).astype(str) + "%")
    return out_df


def main():
    df_meta = pd.read_csv(QUERIES_METADATA)
    df_trans = pd.read_csv(TRANSLATIONS_PATH)

    run_jalur1(df_meta, df_trans)
    run_jalur2(df_meta)


if __name__ == "__main__":
    main()