"""
Implementasi inti: DTW standar, Sakoe-Chiba Band (fixed & adaptive window),
dan Normalized DTW (NDTW).

Mengacu langsung ke rumus di proposal:
- 2.2.3  : gamma(i,j) = d(q_i, c_j) + min(gamma(i-1,j), gamma(i,j-1), gamma(i-1,j-1))
- 2.2.6  : D_norm(Q,R) = sum(delta(w_k)) / K
- 2.2.7  : |i - j| <= r   (Sakoe-Chiba Band, fixed window)
- Bab III: r_adaptive = |N - M| + m   (kontribusi utama: adaptive window)

Jarak lokal distandarkan ke Euclidean Distance (sesuai kesepakatan
konsistensi metrik di seluruh Bab II).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AlignmentResult:
    """Hasil satu proses penyelarasan DTW/NDTW."""
    cost_matrix: np.ndarray          # gamma(i,j), berisi np.inf di luar band
    path: list                       # daftar (i, j) sepanjang jalur optimal, W = {w_1, ..., w_K}
    raw_distance: float              # gamma(N, M), jarak kumulatif belum dinormalisasi
    normalized_distance: float       # D_norm = raw_distance / K  (2.2.6)
    path_length: int                 # K, panjang jalur
    window_size: int                 # r yang dipakai
    n_frames_query: int               # N
    n_frames_reference: int           # M


def euclidean_local_distance(query_features: np.ndarray, ref_features: np.ndarray) -> np.ndarray:
    """
    Hitung matriks jarak lokal Euclidean d(q_i, c_j) antar seluruh pasangan
    frame, sesuai 2.2.3 dan keputusan standardisasi Euclidean Distance.

    Parameters
    ----------
    query_features : np.ndarray, shape (N, D)
        Fitur akustik ujaran pengguna (mis. output Wav2Vec 2.0, D=768).
    ref_features : np.ndarray, shape (M, D)
        Fitur akustik referensi / acoustic anchor.

    Returns
    -------
    np.ndarray, shape (N, M) — matriks jarak lokal d(q_i, c_j).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  -> lebih cepat dari loop bersarang
    q_sq = np.sum(query_features ** 2, axis=1, keepdims=True)      # (N, 1)
    r_sq = np.sum(ref_features ** 2, axis=1, keepdims=True).T      # (1, M)
    cross = query_features @ ref_features.T                        # (N, M)
    dist_sq = np.maximum(q_sq + r_sq - 2 * cross, 0.0)
    return np.sqrt(dist_sq)


def compute_adaptive_window(n_frames_query: int, n_frames_reference: int, margin: int = 1) -> int:
    """
    Hitung lebar jendela adaptif: r_adaptive = |N - M| + m

    Ini KONTRIBUSI UTAMA skripsi (Bab III 3.2.3 / 3.3.2): lebar jendela
    dihitung ulang per-pasangan-ujaran, bukan nilai tetap (fixed) yang
    sama untuk semua data.

    Parameters
    ----------
    n_frames_query : int
        N, jumlah frame ujaran pengguna (hasil VAD + Wav2Vec 2.0).
    n_frames_reference : int
        M, jumlah frame referensi / acoustic anchor.
    margin : int
        m, margin toleransi tambahan (hyperparameter, ditentukan empiris
        lewat eksperimen -- lihat experiment/tune_margin.py).

    Returns
    -------
    int : r_adaptive, dijamin >= |N-M| (syarat kelayakan jalur, 2.2.7).
    """
    if margin < 0:
        raise ValueError("margin (m) tidak boleh negatif")
    return abs(n_frames_query - n_frames_reference) + margin


def dtw_align(
    query_features: np.ndarray,
    ref_features: np.ndarray,
    window_size: int | None = None,
    normalize: bool = True,
) -> AlignmentResult:
    """
    Jalankan DTW / NDTW dengan Sakoe-Chiba Band.

    Parameters
    ----------
    query_features : np.ndarray, shape (N, D)
    ref_features : np.ndarray, shape (M, D)
    window_size : int or None
        r, lebar jendela Sakoe-Chiba Band.
        - None -> DTW standar TANPA constraint (2.2.3 murni, dipakai
          sebagai baseline paling dasar / Fase 1 di proposal).
        - int  -> Sakoe-Chiba Band dengan lebar window_size (2.2.7).
          Gunakan compute_adaptive_window() untuk mode adaptive,
          atau nilai konstan untuk mode fixed.
    normalize : bool
        Jika True, hitung juga D_norm sesuai 2.2.6 (NDTW).

    Returns
    -------
    AlignmentResult

    Raises
    ------
    ValueError
        Jika window_size diberikan tapi lebih kecil dari |N-M| -- sesuai
        pembuktian matematis di 2.2.7: jalur TIDAK MUNGKIN mencapai titik
        akhir (N, M) jika r < |N-M|. Ini bukan bug, ini validasi wajib.
    """
    n, m = query_features.shape[0], ref_features.shape[0]

    if window_size is not None and window_size < abs(n - m):
        raise ValueError(
            f"window_size (r={window_size}) lebih kecil dari |N-M|={abs(n - m)}. "
            f"Jalur penyelarasan tidak mungkin mencapai titik akhir ({n},{m}) "
            f"secara matematis (lihat 2.2.7). Perbesar r, atau gunakan "
            f"compute_adaptive_window() untuk menghitung r otomatis."
        )

    local_dist = euclidean_local_distance(query_features, ref_features)

    INF = np.inf
    gamma = np.full((n + 1, m + 1), INF)
    gamma[0, 0] = 0.0

    for i in range(1, n + 1):
        if window_size is None:
            j_lo, j_hi = 1, m
        else:
            # |i-j| <= r  ->  i-r <= j <= i+r, dipotong ke rentang valid [1, m]
            j_lo = max(1, i - window_size)
            j_hi = min(m, i + window_size)
        for j in range(j_lo, j_hi + 1):
            cost = local_dist[i - 1, j - 1]
            gamma[i, j] = cost + min(gamma[i - 1, j], gamma[i, j - 1], gamma[i - 1, j - 1])

    if gamma[n, m] == INF:
        raise RuntimeError(
            "Titik akhir tidak terjangkau meski lolos validasi window_size. "
            "Periksa kembali implementasi batas j_lo/j_hi."
        )

    path = _traceback(gamma, n, m, window_size)
    raw_distance = gamma[n, m]
    k = len(path)
    normalized_distance = raw_distance / k if normalize else None

    return AlignmentResult(
        cost_matrix=gamma,
        path=path,
        raw_distance=raw_distance,
        normalized_distance=normalized_distance,
        path_length=k,
        window_size=window_size if window_size is not None else max(n, m),
        n_frames_query=n,
        n_frames_reference=m,
    )


def _traceback(gamma: np.ndarray, n: int, m: int, window_size):
    """Telusuri balik dari (N,M) ke (1,1) untuk merekonstruksi jalur optimal
    W = {w_1, ..., w_K}, mengikuti aturan kontinuitas & monotonisitas (2.2.3)."""
    path = [(n, m)]
    i, j = n, m
    while (i, j) != (1, 1):
        candidates = []
        if i > 1:
            candidates.append(((i - 1, j), gamma[i - 1, j]))
        if j > 1:
            candidates.append(((i, j - 1), gamma[i, j - 1]))
        if i > 1 and j > 1:
            candidates.append(((i - 1, j - 1), gamma[i - 1, j - 1]))
        # pilih tetangga dengan gamma terkecil (jalur asal termurah)
        (i, j), _ = min(candidates, key=lambda c: c[1])
        path.append((i, j))
    path.reverse()
    return path
