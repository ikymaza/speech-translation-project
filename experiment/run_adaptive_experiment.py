"""
Eksperimen ADAPTIVE WINDOW — dibandingkan langsung dengan hasil
run_fixed_experiment.py (Jalur 2).

Untuk tiap sampel, r dihitung otomatis: r = |N-M| + m
Diuji beberapa nilai margin m untuk melihat trade-off antara
"ruang gerak toleransi" vs efisiensi (RTF).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from vad.energy_vad import detect_speech_segments, trim_edges_only
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio
from alignment.ndtw import dtw_align, compute_adaptive_window

# ============ CONFIG ============
QUERIES_DIR = Path("data/queries")
QUERIES_METADATA = Path("data/prepared/queries_metadata.csv")
REFERENCE_FEATURES_PATH = Path("data/acoustic_anchor.npy")
MARGIN_CANDIDATES = [1, 2, 5]   # m yang diuji
OUTPUT_PATH = Path("experiment/results_adaptive_window.csv")
# =================================


def main():
    df_meta = pd.read_csv(QUERIES_METADATA)
    extractor = Wav2Vec2FeatureExtractor()
    ref_features = np.load(REFERENCE_FEATURES_PATH)

    results = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Adaptive"):
        wav_path = QUERIES_DIR / row["file"]
        if not wav_path.exists():
            continue
        signal, sr = load_audio(str(wav_path))
        clean_signal = trim_edges_only(signal, sr)
        if len(clean_signal) == 0:
            continue

        t0 = time.perf_counter()
        query_features = extractor.extract(clean_signal, sr)
        t_extract = time.perf_counter() - t0
        n, m_ref = query_features.shape[0], ref_features.shape[0]

        for margin in MARGIN_CANDIDATES:
            r_adaptive = compute_adaptive_window(n, m_ref, margin=margin)
            t0 = time.perf_counter()
            try:
                result = dtw_align(query_features, ref_features, window_size=r_adaptive)
                t_align = time.perf_counter() - t0
                results.append({
                    "file": row["file"], "N": n, "M": m_ref,
                    "margin": margin, "r_adaptive": r_adaptive,
                    "status": "ok",
                    "normalized_distance": result.normalized_distance,
                    "rtf_partial": (t_extract + t_align) / row["duration_sec"],
                })
            except ValueError as e:
                # Idealnya TIDAK PERNAH terjadi untuk adaptive window,
                # karena r_adaptive selalu >= |N-M|. Kalau ini muncul,
                # ada bug di compute_adaptive_window() -- laporkan!
                results.append({
                    "file": row["file"], "N": n, "M": m_ref,
                    "margin": margin, "r_adaptive": r_adaptive,
                    "status": f"FAILED: {e}",
                    "normalized_distance": None, "rtf_partial": None,
                })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSelesai. Disimpan ke {OUTPUT_PATH}\n")
    print("Ringkasan per margin (m):")
    summary = out_df.groupby("margin").agg(
        tingkat_gagal_persen=("status", lambda s: round((s != "ok").mean() * 100, 2)),
        avg_normalized_distance=("normalized_distance", "mean"),
        avg_rtf_partial=("rtf_partial", "mean"),
    )
    print(summary)

    n_failed_total = (out_df["status"] != "ok").sum()
    if n_failed_total > 0:
        print(f"\n⚠️  PERINGATAN: {n_failed_total} kasus GAGAL pada adaptive window. "
              f"Ini seharusnya TIDAK terjadi -- cek ulang implementasi compute_adaptive_window().")
    else:
        print("\n✅ 0% kegagalan di semua margin -- sesuai ekspektasi (r_adaptive selalu >= |N-M|).")


if __name__ == "__main__":
    main()