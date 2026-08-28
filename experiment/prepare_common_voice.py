"""
Filter Common Voice sesuai kriteria 3.2.1 (WAV 16kHz, durasi 3-15 detik),
lalu konversi dari .mp3 ke .wav 16kHz mono.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# ============ CONFIG ============
TSV_PATH = Path("data/common_voice_id/validated.tsv")
CLIPS_DIR = Path("data/common_voice_id/clips")
OUTPUT_AUDIO_DIR = Path("data/prepared/audio")
OUTPUT_METADATA = Path("data/prepared/metadata.csv")
MIN_DURATION = 3.0
MAX_DURATION = 15.0
MAX_SAMPLES = 350   # ambil sedikit lebih dari 300 untuk cadangan (ada yang mungkin gagal dibaca)
TARGET_SR = 16000
# =================================

def main():
    df = pd.read_csv(TSV_PATH, sep="\t")
    print(f"Total baris di validated.tsv: {len(df)}")
    print(f"Kolom tersedia: {list(df.columns)}")

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    checked = 0

    for _, row in tqdm(df.iterrows(), total=min(len(df), MAX_SAMPLES * 3)):
        if len(results) >= MAX_SAMPLES:
            break
        checked += 1
        if checked > MAX_SAMPLES * 3:  # jaga-jaga supaya tidak looping semua data kalau banyak gagal
            break

        mp3_path = CLIPS_DIR / row["path"]
        if not mp3_path.exists():
            continue

        try:
            signal, sr = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            continue

        duration = len(signal) / TARGET_SR
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            continue

        out_name = mp3_path.stem + ".wav"
        out_path = OUTPUT_AUDIO_DIR / out_name
        sf.write(str(out_path), signal, TARGET_SR)

        results.append({
            "file": out_name,
            "sentence_id": row["sentence"],
            "duration_sec": round(duration, 2),
        })

    metadata = pd.DataFrame(results)
    metadata.to_csv(OUTPUT_METADATA, index=False)
    print(f"\nSelesai. {len(metadata)} file lolos filter (durasi {MIN_DURATION}-{MAX_DURATION} detik).")
    print(f"Disimpan ke: {OUTPUT_AUDIO_DIR} dan {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()