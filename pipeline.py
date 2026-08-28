"""
Pipeline end-to-end untuk SATU pasang (ujaran pengguna, referensi):
VAD -> Wav2Vec 2.0 -> NDTW/Sakoe-Chiba (fixed atau adaptive) -> hasil.

Sesuai alur di 3.3.1:
    Suara(ID) --VAD--> Audio Bersih --Wav2Vec2--> Fitur Laten
              --NDTW--> Fitur Teralinyemen --Transformer--> Teks(EN)

Modul ini mengerjakan sampai tahap "Fitur Teralinyemen". Tahap Transformer
(NMT) TIDAK termasuk di sini -- itu arsitektur terpisah (2.2.8) yang
biasanya dipakai lewat model pretrained/fine-tuned (mis. MarianMT,
NLLB, atau model custom kamu sendiri). Lihat catatan di README.md
bagian "Menyambungkan ke Transformer".

Juga mencatat T_inference (2.2.10 poin 3a) untuk keperluan pengukuran RTF:
T_inference di modul ini = waktu VAD + waktu ekstraksi fitur + waktu NDTW
(BELUM termasuk waktu decoding Transformer -- tambahkan sendiri saat
sudah menyambungkan Transformer).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from vad.energy_vad import detect_speech_segments
from alignment.ndtw import dtw_align, compute_adaptive_window, AlignmentResult


@dataclass
class PipelineTiming:
    t_vad: float = 0.0
    t_feature_extraction: float = 0.0
    t_alignment: float = 0.0

    @property
    def t_inference_partial(self) -> float:
        """Jumlah waktu VAD + ekstraksi fitur + alignment (belum termasuk
        decoding Transformer). Tambahkan waktu Transformer kamu sendiri
        untuk mendapatkan T_inference penuh sesuai 2.2.10."""
        return self.t_vad + self.t_feature_extraction + self.t_alignment


@dataclass
class PipelineResult:
    alignment: AlignmentResult
    timing: PipelineTiming
    audio_duration_sec: float
    speech_segments: list = field(default_factory=list)

    @property
    def rtf_partial(self) -> float:
        """RTF parsial (2.2.10 poin 3a), belum termasuk waktu Transformer.
        RTF = T_inference / T_audio."""
        if self.audio_duration_sec == 0:
            return float("inf")
        return self.timing.t_inference_partial / self.audio_duration_sec


def run_pipeline(
    query_signal: np.ndarray,
    sample_rate: int,
    ref_features: np.ndarray,
    feature_extractor,
    window_mode: str = "adaptive",
    fixed_r: int | None = None,
    margin: int = 1,
) -> PipelineResult:
    """
    Jalankan satu ujaran melalui pipeline VAD -> Wav2Vec2 -> NDTW.

    Parameters
    ----------
    query_signal : np.ndarray
        Sinyal audio mentah (belum dipotong VAD), mono, sample_rate Hz.
    sample_rate : int
    ref_features : np.ndarray, shape (M, D)
        Fitur akustik referensi / acoustic anchor (sudah diekstraksi
        sebelumnya sekali di awal, TIDAK perlu VAD ulang tiap kali --
        lihat build_acoustic_anchors.py).
    feature_extractor : Wav2Vec2FeatureExtractor
        Instance dari features/wav2vec_extractor.py.
    window_mode : "adaptive" | "fixed" | "none"
        - "adaptive" : r dihitung otomatis per-ujaran (kontribusi utama).
        - "fixed"    : r konstan, WAJIB isi fixed_r.
        - "none"     : DTW standar tanpa constraint (baseline paling dasar).
    fixed_r : int
        Wajib diisi jika window_mode="fixed".
    margin : int
        m, dipakai jika window_mode="adaptive". Lihat experiment/tune_margin.py
        untuk cara menentukan nilai ini secara empiris.

    Returns
    -------
    PipelineResult
    """
    timing = PipelineTiming()
    audio_duration = len(query_signal) / sample_rate

# --- 1. VAD (3.2.2) ---
    t0 = time.perf_counter()
    from vad.energy_vad import trim_edges_only
    _, _, segments = detect_speech_segments(query_signal, sample_rate)
    clean_signal = trim_edges_only(query_signal, sample_rate)
    timing.t_vad = time.perf_counter() - t0

    if len(clean_signal) == 0:
        raise ValueError("VAD tidak mendeteksi segmen speech sama sekali pada audio ini.")

    # --- 2. Ekstraksi fitur Wav2Vec 2.0 (3.2.3 awal) ---
    t0 = time.perf_counter()
    query_features = feature_extractor.extract(clean_signal, sample_rate)
    timing.t_feature_extraction = time.perf_counter() - t0

    # --- 3. NDTW / Sakoe-Chiba Band (3.2.3 inti, kontribusi utama) ---
    n, m = query_features.shape[0], ref_features.shape[0]
    if window_mode == "adaptive":
        r = compute_adaptive_window(n, m, margin=margin)
    elif window_mode == "fixed":
        if fixed_r is None:
            raise ValueError("window_mode='fixed' butuh fixed_r diisi.")
        r = fixed_r
    elif window_mode == "none":
        r = None
    else:
        raise ValueError(f"window_mode tidak dikenal: {window_mode}")

    t0 = time.perf_counter()
    alignment = dtw_align(query_features, ref_features, window_size=r, normalize=True)
    timing.t_alignment = time.perf_counter() - t0

    return PipelineResult(
        alignment=alignment,
        timing=timing,
        audio_duration_sec=audio_duration,
        speech_segments=segments,
    )
