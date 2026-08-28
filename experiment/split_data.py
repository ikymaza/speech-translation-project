"""
Pisahkan data hasil prepare_common_voice.py jadi:
- subset untuk Acoustic Anchor (100-150 file, sesuai 3.1.1)
- subset untuk data uji eksperimen (sisanya)
"""
import shutil
import random
from pathlib import Path
import pandas as pd

# ============ CONFIG ============
METADATA_PATH = Path("data/prepared/metadata.csv")
AUDIO_DIR = Path("data/prepared/audio")
ANCHOR_DIR = Path("data/acoustic_anchor_subset")
QUERIES_DIR = Path("data/queries")
N_ANCHOR = 120
RANDOM_SEED = 42
# =================================

def main():
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Total data tersedia: {len(metadata)}")

    random.seed(RANDOM_SEED)
    shuffled = metadata.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    anchor_set = shuffled.iloc[:N_ANCHOR]
    query_set = shuffled.iloc[N_ANCHOR:]

    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in anchor_set.iterrows():
        src = AUDIO_DIR / row["file"]
        dst = ANCHOR_DIR / row["file"]
        if src.exists():
            shutil.copy(src, dst)

    for _, row in query_set.iterrows():
        src = AUDIO_DIR / row["file"]
        dst = QUERIES_DIR / row["file"]
        if src.exists():
            shutil.copy(src, dst)

    query_set.to_csv("data/prepared/queries_metadata.csv", index=False)

    print(f"Acoustic Anchor subset: {len(anchor_set)} file -> {ANCHOR_DIR}")
    print(f"Data uji (queries): {len(query_set)} file -> {QUERIES_DIR}")
    print(f"Metadata data uji disimpan ke: data/prepared/queries_metadata.csv")

if __name__ == "__main__":
    main()