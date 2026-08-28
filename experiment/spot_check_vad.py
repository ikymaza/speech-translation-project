import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad.energy_vad import detect_speech_segments, trim_edges_only
from features.wav2vec_extractor import load_audio
import pandas as pd

QUERIES_DIR = Path("data/queries")
df = pd.read_csv("data/prepared/queries_metadata.csv").head(5)

for _, row in df.iterrows():
    wav_path = QUERIES_DIR / row["file"]
    signal, sr = load_audio(str(wav_path))
    raw_duration = len(signal) / sr

    _, _, segments = detect_speech_segments(signal, sr)
    clean_signal = trim_edges_only(signal, sr)
    clean_duration = len(clean_signal) / sr

    print(f"{row['file']}: durasi asli={raw_duration:.2f}s -> setelah VAD={clean_duration:.2f}s "
          f"({len(segments)} segmen ditemukan) -> tersisa {clean_duration/raw_duration*100:.0f}%")