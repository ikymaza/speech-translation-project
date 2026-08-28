import pandas as pd
import requests
import os

# Path dataset Anda
FOLDER_DATASET = r"data\common_voice_id"
NAMA_FOLDER_AUDIO = "clips"

# Baca test.tsv untuk dapat pasangan file audio + transkrip asli (ground truth)
df_test = pd.read_csv(os.path.join(FOLDER_DATASET, "test.tsv"), sep="\t")

# Ambil 5 sampel acak untuk diuji
sampel = df_test.sample(5, random_state=42)

URL_API = "http://127.0.0.1:5000/translate"

for _, baris in sampel.iterrows():
    path_audio = os.path.join(FOLDER_DATASET, NAMA_FOLDER_AUDIO, baris["path"])
    transkrip_asli = baris["sentence"]

    if not os.path.isfile(path_audio):
        print(f"File tidak ditemukan, dilewati: {path_audio}")
        continue

    with open(path_audio, "rb") as f:
        response = requests.post(URL_API, files={"audio": f}, data={"direction": "id-en"})

    try:
        hasil = response.json()
    except Exception as e:
        print(f"Gagal parse response untuk {baris['path']}: {e}")
        print("Response mentah:", response.text)
        continue

    print("=" * 60)
    print("File audio        :", baris["path"])
    print("Transkrip ASLI     :", transkrip_asli)
    print("Transkrip MODEL    :", hasil.get("id_text", f"[ERROR] {hasil.get('error', 'tidak diketahui')}"))
    print("Terjemahan (EN)    :", hasil.get("en_text", "-"))
    print()

print("Selesai menguji", len(sampel), "sampel.")