"""
Menentukan margin (m) secara empiris pada subset kecil data, bukan tebakan.

Karena WER/BLEU sebenarnya baru bisa dihitung SETELAH tahap Transformer
(decoding teks), skrip ini pakai proxy sementara: normalized_distance
(D_norm dari NDTW, 2.2.6) -- semakin kecil D_norm, semakin baik alignment-nya.
Setelah Transformer kamu siap, GANTI fungsi `score_margin` di bawah supaya
memakai WER asli (lebih akurat, dan itu yang harus dilaporkan di Bab IV),
bukan proxy D_norm ini.

Cara pakai:
    python experiment/tune_margin.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline import run_pipeline
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio

QUERIES_DIR = Path("data/queries_tuning")   # subset kecil (20-30 sampel) untuk tuning
REFERENCE_FEATURES_PATH = Path("data/acoustic_anchor.npy")
MARGIN_CANDIDATES = [1, 2, 3, 4, 5]
MODEL_NAME = "facebook/wav2vec2-base-960h"


def score_margin(margin: int, wav_files, ref_features, extractor) -> dict:
    """Rata-ratakan normalized_distance (proxy) dan RTF untuk satu nilai margin.

    GANTI ISI FUNGSI INI dengan perhitungan WER asli begitu Transformer
    sudah tersambung -- misalnya:
        wer_scores = []
        for wav in wav_files:
            hasil = jalankan_pipeline_lengkap_termasuk_transformer(wav, margin)
            wer_scores.append(jiwer.wer(ground_truth[wav], hasil.teks_prediksi))
        return {"avg_wer": np.mean(wer_scores), ...}
    """
    distances, rtfs, n_failed = [], [], 0
    for wav_path in wav_files:
        signal, sr = load_audio(str(wav_path))
        try:
            result = run_pipeline(
                query_signal=signal, sample_rate=sr, ref_features=ref_features,
                feature_extractor=extractor, window_mode="adaptive", margin=margin,
            )
            distances.append(result.alignment.normalized_distance)
            rtfs.append(result.rtf_partial)
        except ValueError:
            n_failed += 1
    return {
        "margin": margin,
        "avg_normalized_distance_proxy": float(np.mean(distances)) if distances else None,
        "avg_rtf_partial": float(np.mean(rtfs)) if rtfs else None,
        "n_failed": n_failed,
        "n_total": len(wav_files),
    }


def main():
    extractor = Wav2Vec2FeatureExtractor(model_name=MODEL_NAME)
    ref_features = np.load(REFERENCE_FEATURES_PATH)
    wav_files = sorted(QUERIES_DIR.glob("*.wav"))

    if not wav_files:
        print(f"Tidak ada file di {QUERIES_DIR}. Siapkan subset kecil (20-30 sampel) dulu.")
        return

    print(f"Menguji {len(MARGIN_CANDIDATES)} kandidat margin pada {len(wav_files)} sampel...\n")
    results = [score_margin(m, wav_files, ref_features, extractor) for m in MARGIN_CANDIDATES]

    print(f"{'margin':<8}{'avg_D_norm_proxy':<20}{'avg_RTF_partial':<18}{'gagal':<10}")
    for r in results:
        print(f"{r['margin']:<8}{r['avg_normalized_distance_proxy']:<20.4f}"
              f"{r['avg_rtf_partial']:<18.4f}{r['n_failed']}/{r['n_total']}")

    best = min(results, key=lambda r: r["avg_normalized_distance_proxy"])
    print(f"\nMargin dengan D_norm proxy terkecil: m={best['margin']}")
    print("PENTING: verifikasi ulang pilihan ini dengan WER asli begitu Transformer siap, "
          "jangan jadikan proxy ini sebagai angka final di Bab IV.")


if __name__ == "__main__":
    main()
