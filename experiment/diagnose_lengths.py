import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from vad.energy_vad import detect_speech_segments, trim_edges_only
from features.wav2vec_extractor import Wav2Vec2FeatureExtractor, load_audio

QUERIES_DIR = Path("data/queries")
QUERIES_METADATA = Path("data/prepared/queries_metadata.csv")
REFERENCE_FEATURES_PATH = Path("data/acoustic_anchor.npy")

extractor = Wav2Vec2FeatureExtractor()
ref_features = np.load(REFERENCE_FEATURES_PATH)
M = ref_features.shape[0]

df = pd.read_csv(QUERIES_METADATA)
n_values = []
for _, row in df.iterrows():
    wav_path = QUERIES_DIR / row["file"]
    if not wav_path.exists():
        continue
    signal, sr = load_audio(str(wav_path))
    clean_signal = trim_edges_only(signal, sr)
    if len(clean_signal) == 0:
        continue
    features = extractor.extract(clean_signal, sr)
    n_values.append(features.shape[0])

n_values = np.array(n_values)
diffs = np.abs(n_values - M)
print(f"M (referensi) = {M}")
print(f"N (query) -> min={n_values.min()}, median={np.median(n_values):.0f}, max={n_values.max()}")
print(f"|N-M| -> min={diffs.min()}, median={np.median(diffs):.0f}, max={diffs.max()}")
print(f"\nPercentile |N-M|: 25%={np.percentile(diffs,25):.0f}  50%={np.percentile(diffs,50):.0f}  "
      f"75%={np.percentile(diffs,75):.0f}  90%={np.percentile(diffs,90):.0f}")