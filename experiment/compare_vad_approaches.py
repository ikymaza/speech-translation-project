import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from features.wav2vec_extractor import load_audio

QUERIES_DIR = Path("data/queries")
SAMPLE_FILE = "common_voice_id_24015699.wav"  # "Hubungan diplomatik..." -- contoh dengan banyak jeda antar-kata


def detect_and_concatenate(signal, sr, **kwargs):
    """Versi LAMA: potong per-segmen lalu SAMBUNG (concatenated)."""
    from vad.energy_vad import detect_speech_segments
    _, clean_signal, _ = detect_speech_segments(signal, sr, **kwargs)
    return clean_signal


def main():
    from vad.energy_vad import trim_edges_only

    wav_path = QUERIES_DIR / SAMPLE_FILE
    signal, sr = load_audio(str(wav_path))

    old_version = detect_and_concatenate(signal, sr)
    new_version = trim_edges_only(signal, sr)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    t_old = np.arange(len(old_version)) / sr
    axes[0].plot(t_old, old_version, color="#C44E52", linewidth=0.6)
    axes[0].set_title(f"VERSI LAMA (Concatenated) — durasi hasil: {len(old_version)/sr:.2f}s — "
                       f"perhatikan sambungan patah antar-segmen")
    axes[0].set_ylabel("Amplitudo")

    t_new = np.arange(len(new_version)) / sr
    axes[1].plot(t_new, new_version, color="#55A868", linewidth=0.6)
    axes[1].set_title(f"VERSI BARU (Trim Edges) — durasi hasil: {len(new_version)/sr:.2f}s — "
                       f"jeda alami antar-kata tetap dipertahankan")
    axes[1].set_xlabel("Waktu (detik)")
    axes[1].set_ylabel("Amplitudo")

    plt.tight_layout()
    plt.savefig("experiment/fig_perbandingan_vad_metode.png", dpi=150)
    plt.show()
    print("Disimpan: experiment/fig_perbandingan_vad_metode.png")


if __name__ == "__main__":
    main()