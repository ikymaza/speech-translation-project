"""
Mengunduh subset LibriSpeech (test-clean) via streaming untuk membangun
Acoustic Anchor Bahasa Inggris -- dataset publik, tidak memerlukan
autentikasi/token Hugging Face, mengikuti kriteria seleksi yang sama
seperti subbab 3.2.1 (durasi 3-15 detik, format WAV 16kHz mono).

Catatan: dataset ini dipakai sebagai pengganti Common Voice English untuk
fitur ekstensi EN->ID (di luar cakupan formal penelitian, lihat batasan
masalah 1.3 poin 4).
"""

import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/acoustic_anchor_subset_en")
TARGET_COUNT = 150
MIN_DURATION = 3.0
MAX_DURATION = 15.0
SAMPLE_RATE = 16000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Menghubungkan ke LibriSpeech test-clean (streaming)...")
    dataset = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        revision="refs/convert/parquet",
    )

    saved_count = 0
    checked_count = 0

    for item in dataset:
        checked_count += 1
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        duration = len(audio_array) / sr

        if not (MIN_DURATION <= duration <= MAX_DURATION):
            continue

        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
            )

        out_path = OUTPUT_DIR / f"librispeech_en_{saved_count:04d}.wav"
        sf.write(str(out_path), audio_array, SAMPLE_RATE)

        saved_count += 1
        print(f"  [{saved_count}/{TARGET_COUNT}] disimpan: {out_path.name} (durasi={duration:.2f}s)")

        if saved_count >= TARGET_COUNT:
            break

    print(f"\nSelesai. {saved_count} file tersimpan dari {checked_count} sampel diperiksa.")
    print(f"Lokasi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()