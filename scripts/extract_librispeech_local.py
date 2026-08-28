"""
Ekstrak subset dari test-clean.tar.gz (LibriSpeech, diunduh manual dari
openslr.org) untuk membangun Acoustic Anchor Bahasa Inggris.
"""

import tarfile
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

ARCHIVE_PATH = Path("downloads/test-clean.tar.gz")
OUTPUT_DIR = Path("data/acoustic_anchor_subset_en")
TARGET_COUNT = 150
MIN_DURATION = 3.0
MAX_DURATION = 15.0
SAMPLE_RATE = 16000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Membuka arsip {ARCHIVE_PATH}...")
    saved_count = 0

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".flac")]
        print(f"Ditemukan {len(members)} file .flac di arsip.")

        for member in members:
            if saved_count >= TARGET_COUNT:
                break

            f = tar.extractfile(member)
            if f is None:
                continue

            signal, sr = sf.read(f)
            duration = len(signal) / sr

            if not (MIN_DURATION <= duration <= MAX_DURATION):
                continue

            if sr != SAMPLE_RATE:
                signal = librosa.resample(
                    signal.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
                )

            out_path = OUTPUT_DIR / f"librispeech_en_{saved_count:04d}.wav"
            sf.write(str(out_path), signal, SAMPLE_RATE)

            saved_count += 1
            print(f"  [{saved_count}/{TARGET_COUNT}] disimpan: {out_path.name} (durasi={duration:.2f}s)")

    print(f"\nSelesai. {saved_count} file tersimpan di {OUTPUT_DIR}")


if __name__ == "__main__":
    main()