# Kode Pendukung: NDTW Adaptive Window Size untuk Speech Translation

Implementasi kontribusi utama skripsi: mekanisme *adaptive window size*
pada Sakoe-Chiba Band dalam Normalized DTW (NDTW), dibandingkan terhadap
*fixed window* sebagai *baseline*.

## Status pengujian

- ✅ `alignment/ndtw.py` (DTW, NDTW, Sakoe-Chiba Band, adaptive window) —
  **sudah diuji dengan data sintetis**, semua kasus lolos: validasi
  `r >= |N-M|`, boundary condition, konsistensi jarak dengan/tanpa constraint.
- ✅ `vad/energy_vad.py` (VAD berbasis energi) — **sudah diuji**, berhasil
  memisahkan segmen speech dari silence pada sinyal sintetis.
- ⚠️ `features/wav2vec_extractor.py` — **belum diuji di sini** karena
  environment pembuatan kode ini tidak punya akses internet ke
  HuggingFace Hub (untuk mengunduh bobot model pretrained). Jalankan
  bagian ini di komputer/Colab kamu sendiri.
- ⚠️ Integrasi Transformer (NMT) — **belum ada di kode ini**. Ini
  arsitektur terpisah (2.2.8), biasanya pakai model pretrained (mis.
  MarianMT/NLLB) atau model kamu sendiri. Lihat bagian "Menyambungkan ke
  Transformer" di bawah.

## Instalasi

```bash
pip install numpy scipy librosa soundfile jiwer torch transformers
```

`numpy`, `scipy` sudah cukup untuk modul `alignment/` dan `vad/`.
`torch`, `transformers`, `librosa`, `soundfile` dibutuhkan untuk
`features/wav2vec_extractor.py` (ekstraksi fitur asli).

## Urutan pemakaian (mengikuti roadmap yang sudah didiskusikan)

### Langkah 1 — Uji pipeline dasar dulu (cek angka N wajar)

```python
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio

extractor = Wav2Vec2FeatureExtractor()
signal, sr = load_audio("contoh.wav")
features = extractor.extract(signal, sr)
print(features.shape)  # (T, 768) -- cek T sesuai ekspektasi ~50 frame/detik
```

### Langkah 2 — Bangun Acoustic Anchor dari TITML-IDN

Isi `data/titml_idn_subset/` dengan 100-150 file `.wav` (sesuai revisi
volume dataset 3.1.1), lalu:

```bash
python experiment/build_acoustic_anchors.py
```

Menghasilkan `data/acoustic_anchor.npy` — dipakai sebagai referensi $M$
di semua eksperimen berikutnya.

### Langkah 3 — Tentukan margin $m$ secara empiris

Isi `data/queries_tuning/` dengan 20-30 sampel kecil, lalu:

```bash
python experiment/tune_margin.py
```

**Penting:** skrip ini pakai proxy `normalized_distance` (bukan WER asli)
karena Transformer belum tersambung. Setelah Transformer siap, ganti
fungsi `score_margin()` di file ini supaya memakai WER sungguhan — itu
yang harus dilaporkan di Bab IV, bukan angka proxy.

### Langkah 4 — Jalankan eksperimen penuh (Skenario A/B/C, 3.4.1)

Isi `data/queries/` dengan seluruh sampel uji (200-300 sesuai revisi
volume dataset), atur `FIXED_R_CANDIDATES` dan `ADAPTIVE_MARGIN` di
`experiment/run_experiment.py` sesuai hasil Langkah 3, lalu:

```bash
python experiment/run_experiment.py
```

Menghasilkan `experiment/results.csv` — satu baris per (sampel × mode),
berisi $N$, $M$, $r$, jarak ternormalisasi, dan komponen waktu (untuk RTF).
**Baris dengan `status` berisi `FAILED: ...` itu bukan bug** — itu kasus
*fixed window* gagal karena $r < |N-M|$, justru bukti empiris untuk
argumen Bab I/II kamu soal kelemahan *fixed window*. Catat berapa
persen kegagalan ini per nilai $r$ untuk pembahasan di Bab IV.

## Menyambungkan ke Transformer (belum ada di kode ini)

Setelah `pipeline.py` menghasilkan fitur teralinyemen
(`result.alignment.path`, bisa dipakai untuk mem-*warp* ulang matriks
fitur query mengikuti jalur optimal), langkah berikutnya:

1. Pilih model Transformer NMT (pretrained atau fine-tuning sendiri).
2. Feed fitur teralinyemen sebagai input Encoder (atau, jika modelmu
   berbasis teks, tambahkan dulu tahap ASR/CTC head di atas fitur
   teralinyemen untuk menghasilkan teks Bahasa Indonesia, baru
   diterjemahkan).
3. Hitung WER (bandingkan teks ASR vs ground truth) dan BLEU (bandingkan
   teks terjemahan vs ground truth) pakai library `jiwer` dan
   `sacrebleu` atau `nltk.translate.bleu_score`.
4. Tambahkan waktu decoding Transformer ke `PipelineTiming` supaya
   `rtf_partial` menjadi RTF penuh sesuai 2.2.10.

Ini bagian yang **paling banyak makan waktu** di roadmap kamu — sisihkan
waktu cukup di Tahap 4 Gantt Chart (Integrasi Transformer & Web).

## Struktur folder

```
ndtw_project/
├── vad/energy_vad.py              # VAD berbasis energi (2.2.2, 3.2.2) -- sudah diuji
├── features/wav2vec_extractor.py  # Ekstraksi Wav2Vec2 (2.2.4/2.2.5) -- perlu internet
├── alignment/ndtw.py              # DTW/NDTW/adaptive window (2.2.3/2.2.6/2.2.7) -- sudah diuji
├── pipeline.py                    # Orkestrasi satu ujaran end-to-end
├── experiment/
│   ├── build_acoustic_anchors.py  # Bangun referensi dari TITML-IDN
│   ├── tune_margin.py             # Tentukan margin m empiris
│   └── run_experiment.py          # Eksperimen batch fixed vs adaptive
└── data/                          # taruh dataset kamu di sini (tidak di-commit)
```
