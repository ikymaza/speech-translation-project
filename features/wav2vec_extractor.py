"""
Ekstraksi fitur akustik menggunakan Wav2Vec 2.0 (2.2.4, 2.2.5).

CATATAN PENTING:
Modul ini butuh `torch` dan `transformers`, dan mengunduh bobot model
pretrained dari HuggingFace Hub saat pertama kali dijalankan -- karena itu
modul ini TIDAK dijalankan di lingkungan pembuatan kode ini (tidak ada
akses internet ke huggingface.co). Jalankan file ini di komputer/Colab
kamu sendiri yang punya akses internet, idealnya dengan GPU.

Install dulu:
    pip install torch transformers librosa soundfile

Model yang dipakai: "facebook/wav2vec2-base-960h" (varian Base, D=768,
sesuai 2.2.5). Untuk performa lebih baik pada Bahasa Indonesia, bisa
diganti ke model yang di-finetune Bahasa Indonesia, mis.
"indonesian-nlp/wav2vec2-large-xlsr-56-indonesian" (cek dimensi hidden
size-nya, kalau beda dari 768 sesuaikan penjelasan di 2.2.5 proposal).
"""

import numpy as np


class Wav2Vec2FeatureExtractor:
    """Wrapper Wav2Vec 2.0 untuk ekstraksi fitur akustik (2.2.4, 2.2.5)."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = "cpu"):
        # Import di dalam __init__, bukan di top-level file, supaya modul
        # lain (mis. alignment/ndtw.py) tetap bisa dites TANPA perlu
        # install torch/transformers sama sekali.
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor as HFExtractor

        self.device = device
        self.processor = HFExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        self._torch = torch

    def extract(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Ekstrak fitur akustik dari sinyal audio bersih (hasil VAD).

        Parameters
        ----------
        signal : np.ndarray
            Sinyal audio mono, sample_rate 16000 Hz (sesuai 3.2.1).
        sample_rate : int
            Wajib 16000 sesuai spesifikasi Wav2Vec 2.0 pretrained.

        Returns
        -------
        np.ndarray, shape (T, 768)
            Matriks fitur laten T x D sesuai 2.2.5. T ~= 50 frame per
            detik audio (rasio khas Wav2Vec 2.0, lihat 2.2.5).
        """
        if sample_rate != 16000:
            raise ValueError(
                f"Wav2Vec 2.0 pretrained butuh sample_rate=16000, dapat {sample_rate}. "
                f"Resample dulu (mis. librosa.resample)."
            )
        inputs = self.processor(signal, sampling_rate=sample_rate, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device))
        # outputs.last_hidden_state shape: (1, T, 768)
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return features


def load_audio(path: str, target_sr: int = 16000) -> tuple:
    """Muat file audio dan resample ke target_sr (sesuai 3.2.1: WAV 16kHz)."""
    import librosa
    signal, sr = librosa.load(path, sr=target_sr, mono=True)
    return signal, sr
