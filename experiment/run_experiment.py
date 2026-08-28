"""
Eksperimen batch: bandingkan fixed window vs adaptive window pada banyak
pasang (ujaran, referensi), sesuai Skenario A/B/C di proposal 3.4.1.

Cara pakai (isi bagian CONFIG di bawah, lalu jalankan):
    python experiment/run_experiment.py

Output: file CSV berisi satu baris per (sampel x mode), siap dianalisis
langsung untuk tabel/grafik di Bab IV (WER/BLEU perlu ditambahkan setelah
Transformer decoding disambungkan -- lihat README.md).

Struktur folder data yang diasumsikan:
    data/
        queries/         # ujaran pengguna (WAV 16kHz), nama file bebas
            u001.wav
            u002.wav
            ...
        references.json  # {"anchor_id": "path/to/anchor_features.npy", ...}
                          # ATAU langsung acoustic_anchor.npy tunggal jika
                          # pakai 1 referensi global (lihat build_acoustic_anchors.py)
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline import run_pipeline
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio


# ============ CONFIG -- SESUAIKAN DENGAN DATASET KAMU ============
QUERIES_DIR = Path("data/queries")            # folder berisi file .wav ujaran
REFERENCE_FEATURES_PATH = Path("data/acoustic_anchor.npy")  # hasil build_acoustic_anchors.py
OUTPUT_CSV = Path("experiment/results.csv")
FIXED_R_CANDIDATES = [3, 4, 5]                  # nilai r fixed yang ingin dibandingkan
ADAPTIVE_MARGIN = 1                             # m, hasil tuning (lihat tune_margin.py)
MODEL_NAME = "facebook/wav2vec2-base-960h"      # ganti ke model B.Indonesia jika ada
# ===================================================================


def main():
    extractor = Wav2Vec2FeatureExtractor(model_name=MODEL_NAME)
    ref_features = np.load(REFERENCE_FEATURES_PATH)

    wav_files = sorted(QUERIES_DIR.glob("*.wav"))
    if not wav_files:
        print(f"Tidak ada file .wav ditemukan di {QUERIES_DIR}. Cek CONFIG di atas.")
        return

    rows = []
    modes = [("adaptive", None)] + [("fixed", r) for r in FIXED_R_CANDIDATES]

    for wav_path in wav_files:
        signal, sr = load_audio(str(wav_path))
        for mode, r in modes:
            try:
                result = run_pipeline(
                    query_signal=signal,
                    sample_rate=sr,
                    ref_features=ref_features,
                    feature_extractor=extractor,
                    window_mode=mode,
                    fixed_r=r,
                    margin=ADAPTIVE_MARGIN,
                )
                rows.append({
                    "file": wav_path.name,
                    "mode": mode if mode == "adaptive" else f"fixed_r{r}",
                    "N": result.alignment.n_frames_query,
                    "M": result.alignment.n_frames_reference,
                    "r_used": result.alignment.window_size,
                    "raw_distance": result.alignment.raw_distance,
                    "normalized_distance": result.alignment.normalized_distance,
                    "path_length_K": result.alignment.path_length,
                    "audio_duration_sec": result.audio_duration_sec,
                    "t_vad": result.timing.t_vad,
                    "t_feature_extraction": result.timing.t_feature_extraction,
                    "t_alignment": result.timing.t_alignment,
                    "t_inference_partial": result.timing.t_inference_partial,
                    "rtf_partial": result.rtf_partial,
                    "status": "ok",
                })
                print(f"[OK] {wav_path.name} mode={mode} r={result.alignment.window_size} "
                      f"D_norm={result.alignment.normalized_distance:.4f} RTF~{result.rtf_partial:.3f}")
            except ValueError as e:
                # Ini yang PALING PENTING untuk skripsi kamu: kasus fixed
                # window gagal karena r < |N-M|. JANGAN skip diam-diam --
                # catat sebagai kegagalan, ini justru bukti empiris klaim
                # "fixed window tidak selalu valid" di Bab I/II.
                rows.append({
                    "file": wav_path.name,
                    "mode": mode if mode == "adaptive" else f"fixed_r{r}",
                    "N": None, "M": None, "r_used": r,
                    "raw_distance": None, "normalized_distance": None,
                    "path_length_K": None, "audio_duration_sec": None,
                    "t_vad": None, "t_feature_extraction": None, "t_alignment": None,
                    "t_inference_partial": None, "rtf_partial": None,
                    "status": f"FAILED: {e}",
                })
                print(f"[GAGAL] {wav_path.name} mode={mode} r={r} -> {e}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSelesai. {len(rows)} baris ditulis ke {OUTPUT_CSV}")

    n_failed = sum(1 for r in rows if r["status"] != "ok")
    print(f"Ringkasan: {n_failed} dari {len(rows)} kombinasi (sampel x mode) GAGAL "
          f"(kemungkinan besar fixed window dengan r terlalu kecil).")


if __name__ == "__main__":
    main()
