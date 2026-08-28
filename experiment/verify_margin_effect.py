import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from vad.energy_vad import trim_edges_only
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio
from alignment.ndtw import dtw_align, compute_adaptive_window

QUERIES_DIR = Path("data/queries")
df_meta = pd.read_csv("data/prepared/queries_metadata.csv")
ref_features = np.load("data/acoustic_anchor.npy")
extractor = Wav2Vec2FeatureExtractor()

# Ambil 1 sampel dengan |N-M| PALING BESAR (kasus paling ekstrem dari diagnose_lengths.py)
print("Mencari sampel dengan N terjauh dari M...")
best_row, best_diff, best_feat = None, -1, None
print(f"Memeriksa seluruh {len(df_meta)} sampel...")
for _, row in df_meta.iterrows():  # cek SEMUA sampel untuk cari kasus ekstrem yang sesungguhnya
    wav_path = QUERIES_DIR / row["file"]
    if not wav_path.exists():
        continue
    signal, sr = load_audio(str(wav_path))
    clean_signal = trim_edges_only(signal, sr)
    if len(clean_signal) == 0:
        continue
    feat = extractor.extract(clean_signal, sr)
    diff = abs(feat.shape[0] - ref_features.shape[0])
    if diff > best_diff:
        best_diff, best_row, best_feat = diff, row, feat

print(f"Sampel terpilih: {best_row['file']}, N={best_feat.shape[0]}, M={ref_features.shape[0]}, |N-M|={best_diff}")

print("\nUji berbagai margin pada sampel INI:")
for margin in [1, 2, 5, 20, 50, 100]:
    r = compute_adaptive_window(best_feat.shape[0], ref_features.shape[0], margin=margin)
    result = dtw_align(best_feat, ref_features, window_size=r)
    print(f"margin={margin:3d}  r={r:4d}  normalized_distance={result.normalized_distance:.6f}")
    
    print("\nBandingkan dengan DTW TANPA constraint sama sekali (window_size=None):")
    result_unconstrained = dtw_align(best_feat, ref_features, window_size=None)
    print(f"unconstrained  normalized_distance={result_unconstrained.normalized_distance:.6f}")