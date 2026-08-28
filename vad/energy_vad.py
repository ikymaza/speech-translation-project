"""
Voice Activity Detection (VAD) berbasis energi sinyal.

Pipeline:
Audio
  -> Short-Time Energy
  -> Deteksi speech
  -> Smoothing
  -> Padding awal/akhir
  -> Audio bersih untuk ASR

Catatan:
Implementasi ini tetap merupakan energy-based VAD, bukan deep-learning VAD.

Perbaikan utama:
1. Mempertahankan fungsi trim_edges_only.
2. Menambahkan padding pada awal dan akhir speech.
3. Padding digunakan agar fonem awal/akhir tidak mudah terpotong.
4. Jeda alami di tengah kalimat TIDAK dipotong.
"""

import numpy as np


def _frame_signal(
    signal: np.ndarray,
    frame_len: int,
    hop_len: int
) -> np.ndarray:
    """
    Membagi sinyal 1D menjadi frame overlapping.
    """

    if len(signal) < frame_len:
        signal = np.pad(
            signal,
            (0, frame_len - len(signal))
        )

    n_frames = 1 + (len(signal) - frame_len) // hop_len

    indices = (
        np.arange(frame_len)[None, :]
        + hop_len * np.arange(n_frames)[:, None]
    )

    return signal[indices]


def _short_time_energy(frames: np.ndarray) -> np.ndarray:
    """
    Menghitung short-time energy setiap frame.
    """

    return np.mean(
        frames.astype(np.float64) ** 2,
        axis=1
    )


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """
    Menghitung Zero Crossing Rate (ZCR).

    ZCR dihitung untuk kebutuhan analisis VAD,
    tetapi keputusan speech pada implementasi ini
    tetap berbasis energy threshold.
    """

    signs = np.sign(frames)

    signs[signs == 0] = 1

    return np.mean(
        np.abs(np.diff(signs, axis=1)) > 0,
        axis=1
    )


def _remove_short_runs(
    mask: np.ndarray,
    min_len: int,
    target: bool
) -> np.ndarray:
    """
    Menghapus run pendek dari nilai target.

    Contoh:
    True True False True True

    Jika run True terlalu pendek, akan diubah
    menjadi False.
    """

    mask = mask.copy()

    i = 0
    n = len(mask)

    while i < n:

        if mask[i] == target:

            j = i

            while j < n and mask[j] == target:
                j += 1

            if (j - i) < min_len:
                mask[i:j] = not target

            i = j

        else:
            i += 1

    return mask


def detect_speech_segments(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_threshold_ratio: float = 0.02,
    min_speech_ms: float = 100.0,
    min_silence_ms: float = 250.0,
):
    """
    Deteksi segmen speech menggunakan short-time energy.

    Returns
    -------
    speech_mask :
        Boolean mask setiap frame.

    clean_signal :
        Gabungan seluruh segmen speech.

    segments :
        List tuple:
        (start_second, end_second)
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        return (
            np.array([], dtype=bool),
            np.array([], dtype=np.float32),
            []
        )

    frame_len = max(
        1,
        int(sample_rate * frame_ms / 1000)
    )

    hop_len = max(
        1,
        int(sample_rate * hop_ms / 1000)
    )

    frames = _frame_signal(
        signal,
        frame_len,
        hop_len
    )

    energy = _short_time_energy(frames)

    # Tetap dihitung untuk kebutuhan analisis.
    _ = _zero_crossing_rate(frames)

    max_energy = float(np.max(energy))

    if max_energy <= 0:
        return (
            np.zeros(len(frames), dtype=bool),
            np.array([], dtype=signal.dtype),
            []
        )

    # Noise mikrofon yang konstan memiliki energi yang hampir sama
    # di semua frame. Tolak sinyal seperti ini sebelum threshold relatif
    # dijalankan agar Whisper tidak melakukan hallucination pada silence.
    noise_floor = float(np.percentile(energy, 20))
    active_energy = float(np.percentile(energy, 90))
    if active_energy <= max(noise_floor * 1.8, np.finfo(np.float64).eps):
        return (
            np.zeros(len(frames), dtype=bool),
            np.array([], dtype=signal.dtype),
            []
        )

    # Threshold relatif terhadap energi maksimum.
    energy_threshold = max(
        energy_threshold_ratio * max_energy,
        noise_floor * 1.5,
    )

    is_speech = energy > energy_threshold

    # ---------------------------------------------------------
    # SMOOTHING
    # ---------------------------------------------------------

    min_speech_frames = max(
        1,
        int(np.ceil(min_speech_ms / hop_ms))
    )

    min_silence_frames = max(
        1,
        int(np.ceil(min_silence_ms / hop_ms))
    )

    # Buang speech yang terlalu pendek.
    is_speech = _remove_short_runs(
        is_speech,
        min_speech_frames,
        target=True
    )

    # Tutup silence pendek supaya jeda alami
    # di tengah kalimat tidak dianggap sebagai pemisah.
    is_speech = _remove_short_runs(
        is_speech,
        min_silence_frames,
        target=False
    )

    # ---------------------------------------------------------
    # BANGUN SEGMENT
    # ---------------------------------------------------------

    segments = []

    clean_chunks = []

    start_idx = None

    for i, flag in enumerate(is_speech):

        if flag and start_idx is None:

            start_idx = i

        elif not flag and start_idx is not None:

            seg_start_sample = start_idx * hop_len

            seg_end_sample = min(
                len(signal),
                i * hop_len + frame_len
            )

            if seg_end_sample > seg_start_sample:

                clean_chunks.append(
                    signal[
                        seg_start_sample:seg_end_sample
                    ]
                )

                segments.append(
                    (
                        seg_start_sample / sample_rate,
                        seg_end_sample / sample_rate
                    )
                )

            start_idx = None

    # Jika speech berakhir di frame terakhir.
    if start_idx is not None:

        seg_start_sample = start_idx * hop_len

        seg_end_sample = len(signal)

        if seg_end_sample > seg_start_sample:

            clean_chunks.append(
                signal[
                    seg_start_sample:seg_end_sample
                ]
            )

            segments.append(
                (
                    seg_start_sample / sample_rate,
                    seg_end_sample / sample_rate
                )
            )

    if clean_chunks:

        clean_signal = np.concatenate(
            clean_chunks
        ).astype(signal.dtype)

    else:

        clean_signal = np.array(
            [],
            dtype=signal.dtype
        )

    return (
        is_speech,
        clean_signal,
        segments
    )


def trim_edges_only(
    signal: np.ndarray,
    sample_rate: int,
    padding_ms: float = 150.0,
    **kwargs
):
    """
    Memotong hanya silence pada awal dan akhir audio.

    Jeda alami di tengah kalimat dipertahankan.

    Perbaikan:
    Menambahkan padding temporal pada awal dan akhir
    agar fonem/kata tidak terpotong terlalu agresif.

    Contoh:

        VAD menemukan:
        0.35s -> 2.40s

        padding = 150 ms

        hasil:
        0.20s -> 2.55s
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if signal.size == 0:
        return np.array(
            [],
            dtype=signal.dtype
        )

    _, _, segments = detect_speech_segments(
        signal,
        sample_rate,
        **kwargs
    )

    if not segments:

        return np.array(
            [],
            dtype=signal.dtype
        )

    padding_samples = int(
        sample_rate * padding_ms / 1000
    )

    first_start_sample = int(
        segments[0][0] * sample_rate
    )

    last_end_sample = int(
        segments[-1][1] * sample_rate
    )

    # Padding kiri.
    start_sample = max(
        0,
        first_start_sample - padding_samples
    )

    # Padding kanan.
    end_sample = min(
        len(signal),
        last_end_sample + padding_samples
    )

    return signal[
        start_sample:end_sample
    ]


def get_vad_info(
    signal: np.ndarray,
    sample_rate: int,
    padding_ms: float = 150.0,
    **kwargs
):
    """
    Fungsi tambahan untuk diagnostic.

    Mengembalikan informasi VAD tanpa mengubah
    pipeline utama.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    _, _, segments = detect_speech_segments(
        signal,
        sample_rate,
        **kwargs
    )

    if not segments:

        return {
            "speech_detected": False,
            "segments": [],
            "original_duration_sec": round(
                len(signal) / sample_rate,
                3
            ),
            "vad_duration_sec": 0.0,
            "padding_ms": padding_ms,
        }

    padding_samples = int(
        sample_rate * padding_ms / 1000
    )

    start_sample = max(
        0,
        int(segments[0][0] * sample_rate)
        - padding_samples
    )

    end_sample = min(
        len(signal),
        int(segments[-1][1] * sample_rate)
        + padding_samples
    )

    return {
        "speech_detected": True,
        "segments": [
            {
                "start_sec": round(float(start), 3),
                "end_sec": round(float(end), 3),
            }
            for start, end in segments
        ],
        "original_duration_sec": round(
            len(signal) / sample_rate,
            3
        ),
        "vad_duration_sec": round(
            (end_sample - start_sample) / sample_rate,
            3
        ),
        "trim_start_sec": round(
            start_sample / sample_rate,
            3
        ),
        "trim_end_sec": round(
            end_sample / sample_rate,
            3
        ),
        "padding_ms": padding_ms,
    }