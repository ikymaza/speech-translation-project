"""
Hasilkan referensi terjemahan Inggris dari transkrip Bahasa Indonesia
Common Voice, memakai Google Translate (via deep-translator).

CATATAN PENTING untuk Bab IV nanti:
Referensi BLEU pada penelitian ini dihasilkan dari mesin penerjemah
eksternal (Google Translate), BUKAN terjemahan manusia asli, karena
kendala akses dataset paralel (Google FLEURS gagal diakses, TITML-IDN
tidak berisi rekaman). Ini adalah batasan penelitian yang harus
dicantumkan secara eksplisit di Bab IV.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

# ============ CONFIG ============
QUERIES_METADATA = Path("data/prepared/queries_metadata.csv")
OUTPUT_PATH = Path("data/translations.csv")
# =================================

def main():
    df = pd.read_csv(QUERIES_METADATA)
    print(f"Menerjemahkan {len(df)} kalimat Indonesia -> Inggris...")

    translator = GoogleTranslator(source="id", target="en")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        id_text = str(row["sentence_id"])  # kolom ini berisi teks kalimat ID (lihat catatan penamaan)
        try:
            en_text = translator.translate(id_text)
        except Exception as e:
            print(f"Gagal menerjemahkan '{id_text[:30]}...': {e}")
            en_text = None
        results.append({
            "file": row["file"],
            "id_text": id_text,
            "en_reference": en_text,
        })
        time.sleep(0.3)  # jaga-jaga supaya tidak kena rate limit Google Translate

    out_df = pd.DataFrame(results)
    n_failed = out_df["en_reference"].isna().sum()
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSelesai. {len(out_df)-n_failed}/{len(out_df)} berhasil diterjemahkan.")
    print(f"Disimpan ke: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()