"""
Membangun Acoustic Anchor dari subset LibriSpeech (Bahasa Inggris),
untuk fitur ekstensi EN->ID (di luar cakupan formal penelitian utama,
lihat batasan masalah 1.3 poin 4).

Metodologi identik dengan build_acoustic_anchors.py versi Indonesia:
Centroid Template dari fitur Wav2Vec2, di-resample ke panjang median,
lalu dirata-ratakan jadi satu template referensi tunggal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio
from vad.energy_vad import trim_edges_only

SUBSET_DIR = Path("data/acoustic_anchor_subset_en")
OUTPUT_PATH = Path("data/acoustic_anchor_en.npy")
MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_LENGTH_FRAMES = None  # None = pakai panjang median dari subset


def main():
    extractor = Wav2Vec2FeatureExtractor(model_name=MODEL_NAME)
    wav_files = sorted(SUBSET_DIR.glob("*.wav"))

    if not wav_files:
        print(f"Tidak ada file di {SUBSET_DIR}. Pastikan subset LibriSpeech sudah diekstrak.")
        return

    print(f"Mengekstrak fitur dari {len(wav_files)} file LibriSpeech (EN)...")
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

    resampled = [_resample_frames(f, target_len) for f in all_features]
    anchor = np.mean(np.stack(resampled, axis=0), axis=0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, anchor)
    print(f"\nAcoustic Anchor (EN) tersimpan: {OUTPUT_PATH} shape={anchor.shape} "
          f"(M={anchor.shape[0]} frame, D={anchor.shape[1]})")


def _resample_frames(features: np.ndarray, target_len: int) -> np.ndarray:
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