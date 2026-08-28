"""
Membangun Acoustic Anchor dari subset TITML-IDN (3.1.1 poin 3).

Sesuai proposal: representasi sampel fonem/kluster akustik dominan Bahasa
Indonesia diekstrak fiturnya via Wav2Vec 2.0, lalu dijadikan template
referensi (Centroid Template) -- BUKAN memakai seluruh dataset TITML-IDN
mentah-mentah untuk tiap perbandingan (itu yang justru diperingatkan
sebagai "kebanyakan" oleh penguji, poin revisi 8).

Cara kerja skrip ini (Centroid Template sederhana):
1. Ekstrak fitur Wav2Vec2 dari tiap file di TITML-IDN subset (100-150
   kata/frasa sesuai revisi volume dataset).
2. Kelompokkan berdasarkan kemiripan (di sini: rata-ratakan semua jadi
   SATU template global sederhana; untuk versi lebih canggih bisa pakai
   k-means clustering supaya dapat beberapa "acoustic anchor" mewakili
   kluster fonem berbeda -- lihat catatan di bawah).
3. Simpan hasil ke acoustic_anchor.npy, dipakai run_experiment.py.

Catatan pengembangan lanjut (opsional, kalau punya waktu):
Alih-alih 1 template tunggal, gunakan sklearn.cluster.KMeans pada seluruh
frame gabungan untuk membentuk beberapa "centroid" (mis. 5-10 kluster),
lalu pilih 1 anchor per kluster fonem dominan -- ini lebih dekat dengan
istilah "Centroid Template" yang disebut di 3.1.1, dan hasilnya lebih
representatif dibanding 1 template rata-rata tunggal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio
from vad.energy_vad import trim_edges_only

TITML_SUBSET_DIR = Path("data/acoustic_anchor_subset")
OUTPUT_PATH = Path("data/acoustic_anchor.npy")
MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_LENGTH_FRAMES = None  # None = pakai panjang median dari subset


def main():
    extractor = Wav2Vec2FeatureExtractor(model_name=MODEL_NAME)
    wav_files = sorted(TITML_SUBSET_DIR.glob("*.wav"))

    if not wav_files:
        print(f"Tidak ada file di {TITML_SUBSET_DIR}. Siapkan subset TITML-IDN dulu "
              f"(100-150 file sesuai revisi 3.1.1).")
        return

    print(f"Mengekstrak fitur dari {len(wav_files)} file TITML-IDN...")
    all_features = []
    for wav_path in wav_files:
        signal, sr = load_audio(str(wav_path))
        clean_signal = trim_edges_only(signal, sr)
        if len(clean_signal) == 0:
            print(f"  {wav_path.name}: SKIP (VAD tidak deteksi speech)")
            continue
        feats = extractor.extract(clean_signal, sr)
        all_features.append(feats)
        print(f"  {wav_path.name}: {feats.shape[0]} frame")

    lengths = [f.shape[0] for f in all_features]
    target_len = TARGET_LENGTH_FRAMES or int(np.median(lengths))
    print(f"\nPanjang frame subset: min={min(lengths)}, median={target_len}, max={max(lengths)}")

    # Samakan panjang tiap fitur ke target_len via interpolasi linear
    # sederhana per-dimensi, lalu rata-ratakan jadi satu Centroid Template.
    resampled = [_resample_frames(f, target_len) for f in all_features]
    anchor = np.mean(np.stack(resampled, axis=0), axis=0)  # (target_len, D)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, anchor)
    print(f"\nAcoustic Anchor tersimpan: {OUTPUT_PATH} shape={anchor.shape} "
          f"(M={anchor.shape[0]} frame, D={anchor.shape[1]})")


def _resample_frames(features: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolasi linear sepanjang sumbu waktu (axis=0) ke target_len frame."""
    n, d = features.shape
    if n == target_len:
        return features
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, d))
    for dim in range(d):
        resampled[:, dim] = np.interp(x_new, x_old, features[:, dim])
    return resampled


if __name__ == "__main__":
    main()
