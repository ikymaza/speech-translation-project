"""
Visualisasi VAD: plot waveform + area yang dianggap speech vs silence,
supaya bisa DILIHAT LANGSUNG apakah VAD memotong ucapan asli atau tidak.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vad.energy_vad import detect_speech_segments
from features.wav2vec_extractor import load_audio

QUERIES_DIR = Path("data/queries")
OUTPUT_DIR = Path("experiment/vad_diagnosis")
N_SAMPLES = 5

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/prepared/queries_metadata.csv").head(N_SAMPLES)

    for _, row in df.iterrows():
        wav_path = QUERIES_DIR / row["file"]
        signal, sr = load_audio(str(wav_path))
        mask, clean_signal, segments = detect_speech_segments(signal, sr)

        time_axis = np.arange(len(signal)) / sr

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_axis, signal, color="gray", linewidth=0.5, label="Sinyal audio")

        for start, end in segments:
            ax.axvspan(start, end, color="green", alpha=0.3)

        ax.set_title(f"{row['file']}  |  \"{row['sentence_id']}\"  |  "
                      f"durasi asli={len(signal)/sr:.2f}s, tersisa={len(clean_signal)/sr:.2f}s "
                      f"({len(clean_signal)/len(signal)*100:.0f}%)")
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Amplitudo")
        ax.legend(["Sinyal audio", "Terdeteksi SPEECH (area hijau)"])

        out_path = OUTPUT_DIR / f"{row['file']}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"Disimpan: {out_path}")

    print(f"\nBuka folder {OUTPUT_DIR} dan lihat gambar-gambarnya. "
          f"Area HIJAU = dianggap speech oleh VAD. "
          f"Cek apakah area PUTIH (non-hijau) itu benar-benar hening, "
          f"atau ternyata masih ada suara/ucapan di situ.")

if __name__ == "__main__":
    main()