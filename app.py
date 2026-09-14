"""
Backend Flask Speech Translation.

Pipeline:

Audio
  |
  v
VAD Energy + Padding
  |
  +------------------------------+
  |                              |
  v                              v
Whisper Fine-tuned           Wav2Vec2
ASR                          Feature Extraction
  |                              |
  v                              v
Google Translate             Adaptive NDTW
(deep-translator)
  |                              |
  v                              v
Teks Terjemahan              Statistik Teknis
  |
  v
gTTS

Catatan:
- Whisper menghasilkan teks yang ditampilkan kepada pengguna.
- Wav2Vec2 + NDTW hanya digunakan untuk statistik/analisis teknis.
- NDTW TIDAK mengubah hasil Whisper.
- Penerjemahan pakai Google Translate (via deep-translator, tanpa API
  key) - MarianMT lokal sudah tidak dipakai lagi.
"""

import sys
import time
import uuid
import re
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)


# ============================================================
# IMPORT
# ============================================================

import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)

from flask_cors import CORS

from gtts import gTTS

from transformers import (
    pipeline as hf_pipeline,
    MarianMTModel,
    MarianTokenizer,
)

from vad.energy_vad import (
    trim_edges_only,
    get_vad_info,
)

from features.wav2vec_extractor import (
    Wav2Vec2FeatureExtractor,
)

from alignment.ndtw import (
    dtw_align,
    compute_adaptive_window,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

CORS(app)


# ============================================================
# DIRECTORY
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

AUDIO_OUTPUT_DIR = (
    BASE_DIR
    / "static"
    / "audio_output"
)

AUDIO_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

# CPU karena RAM terbatas.
DEVICE = -1


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# True = menggunakan Whisper hasil fine-tuning.
# False = menggunakan Whisper base.
#
# Jangan diganti lewat frontend.
# Restart server jika ingin mengganti.
PAKAI_MODEL_FINETUNED = True


PATH_MODEL_FINETUNED = (
    BASE_DIR
    / "model"
    / "whisper_finetuned"
)

MODEL_WHISPER_DASAR = (
    "openai/whisper-small"
)


# ============================================================
# WHISPER GENERATION
# ============================================================

# Sengaja dibuat minimal untuk pengujian ASR.
#
# Tidak menggunakan:
# - repetition_penalty
# - no_repeat_ngram_size
#
# karena kita ingin mengukur kemampuan model secara lebih natural.
GENERATE_KWARGS_ASR = {
    "condition_on_prev_tokens": False,
    "temperature": 0.0,
}


# ============================================================
# VAD CONFIGURATION
# ============================================================

VAD_PADDING_MS = 150.0

VAD_ENERGY_THRESHOLD_RATIO = 0.02

VAD_MIN_SPEECH_MS = 100.0

VAD_MIN_SILENCE_MS = 250.0


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("MEMUAT MODEL SPEECH TRANSLATION")
print("=" * 70)
print()


# ============================================================
# 1. WHISPER
# ============================================================

print("[1/3] Memuat Whisper...")


if PAKAI_MODEL_FINETUNED:

    if not PATH_MODEL_FINETUNED.exists():

        raise FileNotFoundError(
            "Model fine-tuned tidak ditemukan:\n"
            f"{PATH_MODEL_FINETUNED}"
        )

    from transformers import (
        WhisperTokenizerFast,
        WhisperFeatureExtractor,
    )

    tokenizer_asr = (
        WhisperTokenizerFast.from_pretrained(
            MODEL_WHISPER_DASAR,
            language="indonesian",
            task="transcribe",
        )
    )

    tokenizer_asr_en = (
        WhisperTokenizerFast.from_pretrained(
            MODEL_WHISPER_DASAR,
            language="english",
            task="transcribe",
        )
    )

    feature_extractor_asr = (
        WhisperFeatureExtractor.from_pretrained(
            MODEL_WHISPER_DASAR
        )
    )

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=str(PATH_MODEL_FINETUNED),
        tokenizer=tokenizer_asr,
        feature_extractor=feature_extractor_asr,
        device=DEVICE,
        generate_kwargs=GENERATE_KWARGS_ASR,
    )

    asr_en = hf_pipeline(
        "automatic-speech-recognition",
        model=MODEL_WHISPER_DASAR,
        tokenizer=tokenizer_asr_en,
        feature_extractor=feature_extractor_asr,
        device=DEVICE,
        generate_kwargs=GENERATE_KWARGS_ASR,
    )

else:

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=MODEL_WHISPER_DASAR,
        device=DEVICE,
        generate_kwargs=GENERATE_KWARGS_ASR,
    )

    asr_en = asr


print(
    "[1/3] Whisper selesai."
)

print(
    "      Mode:",
    "FINE-TUNED"
    if PAKAI_MODEL_FINETUNED
    else "BASELINE"
)

print()


# ============================================================
# 2. MARIANMT (CADANGAN DARURAT — bukan mesin utama)
# ============================================================
#
# Mesin utama penerjemahan tetap Google Translate (deep-translator).
# MarianMT dimuat di sini HANYA sebagai jaring pengaman kalau
# GoogleTranslator dan MyMemoryTranslator dua-duanya gagal (mis. tidak
# ada internet sama sekali saat demo/sidang). Kualitasnya untuk nama
# tempat/entitas lebih lemah dibanding Google Translate, tapi tidak
# pernah gagal karena berjalan 100% lokal tanpa internet.

print("[2/3] Memuat MarianMT (cadangan darurat)...")


print(
    "      Memuat ID -> EN..."
)

mt_tokenizer_id_en = (
    MarianTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-id-en"
    )
)

mt_model_id_en = (
    MarianMTModel.from_pretrained(
        "Helsinki-NLP/opus-mt-id-en"
    )
)

mt_model_id_en.eval()


print(
    "      Memuat EN -> ID..."
)

mt_tokenizer_en_id = (
    MarianTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-id"
    )
)

mt_model_en_id = (
    MarianMTModel.from_pretrained(
        "Helsinki-NLP/opus-mt-en-id"
    )
)

mt_model_en_id.eval()


print(
    "[2/3] MarianMT (cadangan darurat) selesai."
)

print()


# ============================================================
# 3. WAV2VEC2 + ACOUSTIC ANCHOR
# ============================================================

print("[3/3] Memuat Wav2Vec2...")


wav2vec_extractor = (
    Wav2Vec2FeatureExtractor()
)


ACOUSTIC_ANCHOR_ID_PATH = (
    DATA_DIR / "acoustic_anchor.npy"
)

ACOUSTIC_ANCHOR_EN_PATH = (
    DATA_DIR / "acoustic_anchor_en.npy"
)


if not ACOUSTIC_ANCHOR_ID_PATH.exists():

    raise FileNotFoundError(
        "acoustic_anchor.npy tidak ditemukan:\n"
        f"{ACOUSTIC_ANCHOR_ID_PATH}"
    )


if not ACOUSTIC_ANCHOR_EN_PATH.exists():

    raise FileNotFoundError(
        "acoustic_anchor_en.npy tidak ditemukan:\n"
        f"{ACOUSTIC_ANCHOR_EN_PATH}"
    )


acoustic_anchor_id = np.load(
    ACOUSTIC_ANCHOR_ID_PATH
)

acoustic_anchor_en = np.load(
    ACOUSTIC_ANCHOR_EN_PATH
)


print(
    "      Anchor ID shape:",
    acoustic_anchor_id.shape
)

print(
    "      Anchor EN shape:",
    acoustic_anchor_en.shape
)

print(
    "[3/3] Wav2Vec2 selesai."
)

print()
print("=" * 70)
print("SEMUA MODEL BERHASIL DIMUAT")
print("=" * 70)
print()


# ============================================================
# TRANSLATION — 3 lapis fallback
# ============================================================
#
# Lapis 1: GoogleTranslator (deep-translator, tanpa API key) - kualitas
#          terbaik, termasuk soal nama tempat/entitas. Endpoint publik
#          translate.google.com, jadi bisa kena rate-limit (maks 5
#          request/detik menurut Google) kalau testing terlalu cepat
#          beruntun, atau gagal kalau internet mati.
# Lapis 2: MyMemoryTranslator (deep-translator, tanpa API key) - dicoba
#          kalau Lapis 1 gagal. Pakai NAMA BAHASA LENGKAP ("indonesian"/
#          "english"), bukan kode singkat ("id"/"en") - versi kode
#          singkat sempat menyebabkan error "No support for the
#          provided language" pada percobaan sebelumnya.
# Lapis 3: MarianMT LOKAL (Helsinki-NLP/opus-mt) - jaring pengaman
#          darurat kalau Lapis 1 DAN 2 dua-duanya gagal (mis. tidak ada
#          internet sama sekali). Tidak akurat soal nama tempat/entitas
#          (lihat riwayat pengujian sebelumnya), tapi tidak pernah gagal
#          karena berjalan 100% lokal tanpa internet sama sekali.

from deep_translator import GoogleTranslator, MyMemoryTranslator

_TRANSLATE_RETRY_ATTEMPTS = 2
_TRANSLATE_RETRY_DELAY_SEC = 1.5

_MYMEMORY_LANG_NAMES = {
    "id": "indonesian",
    "en": "english",
}


def _translate_with_marianmt_fallback(
    source_text: str,
    direction: str
) -> str:
    if direction == "id-en":
        inputs = mt_tokenizer_id_en(
            source_text,
            return_tensors="pt",
            padding=True,
        )
        output = mt_model_id_en.generate(**inputs)
        return mt_tokenizer_id_en.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

    inputs = mt_tokenizer_en_id(
        source_text,
        return_tensors="pt",
        padding=True,
    )
    output = mt_model_en_id.generate(**inputs)
    return mt_tokenizer_en_id.decode(
        output[0],
        skip_special_tokens=True,
    ).strip()


def _translate_text(
    source_text: str,
    direction: str
) -> str:

    source_text = (
        source_text or ""
    ).strip()

    if not source_text:
        return ""

    source_lang = "id" if direction == "id-en" else "en"
    target_lang = "en" if direction == "id-en" else "id"

    # ----- Lapis 1: GoogleTranslator (dengan retry) -----
    for attempt in range(1, _TRANSLATE_RETRY_ATTEMPTS + 1):
        try:
            translated = GoogleTranslator(
                source=source_lang,
                target=target_lang,
            ).translate(source_text)

            if translated:
                return translated.strip()

        except Exception as e:
            print(
                f"[Lapis 1: GoogleTranslator gagal, percobaan "
                f"{attempt}/{_TRANSLATE_RETRY_ATTEMPTS}] {e}"
            )
            if attempt < _TRANSLATE_RETRY_ATTEMPTS:
                time.sleep(_TRANSLATE_RETRY_DELAY_SEC)

    # ----- Lapis 2: MyMemoryTranslator (nama bahasa lengkap) -----
    try:
        translated = MyMemoryTranslator(
            source=_MYMEMORY_LANG_NAMES[source_lang],
            target=_MYMEMORY_LANG_NAMES[target_lang],
        ).translate(source_text)

        if translated:
            print("[Lapis 2: fallback ke MyMemoryTranslator berhasil]")
            return translated.strip()

    except Exception as e:
        print(f"[Lapis 2: MyMemoryTranslator juga gagal] {e}")

    # ----- Lapis 3: MarianMT lokal (jaring pengaman darurat) -----
    try:
        translated = _translate_with_marianmt_fallback(
            source_text,
            direction,
        )
        print(
            "[Lapis 3: fallback DARURAT ke MarianMT lokal berhasil - "
            "kualitas nama tempat/entitas mungkin kurang akurat]"
        )
        return translated

    except Exception as e:
        print(f"[Lapis 3: MarianMT lokal juga gagal] {e}")

    return (
        "(Terjemahan gagal - coba rekam ulang beberapa saat lagi)"
    )


# ============================================================
# TTS
# ============================================================

def _generate_tts(
    text: str,
    direction: str
) -> str:

    audio_filename = (
        f"{uuid.uuid4().hex}.mp3"
    )

    audio_output_path = (
        AUDIO_OUTPUT_DIR
        / audio_filename
    )

    tts_lang = (
        "en"
        if direction == "id-en"
        else "id"
    )

    gTTS(
        text=text,
        lang=tts_lang,
    ).save(
        str(audio_output_path)
    )

    return audio_filename


# ============================================================
# FILTER HALUSINASI WHISPER PADA AUDIO HENING/NOISE
# ============================================================
#
# Whisper (termasuk model base "openai/whisper-small" yang dipakai untuk
# arah en-id) dikenal luas kadang "menghalusinasi" kalimat penutup video
# YouTube (mis. "Thank you for watching...") saat diberi audio yang
# nyaris hening / hanya noise ambient (kipas, dengung mikrofon, dll),
# karena model dilatih dari sangat banyak video YouTube. Pengecekan
# RMS/peak amplitude di route /translate tidak selalu menangkap ini,
# karena noise ambient bisa saja tetap di atas ambang batas tersebut
# walau bukan ucapan sungguhan.
#
# Mitigasi: cek apakah HASIL TRANSKRIPSI, setelah dinormalisasi (huruf
# kecil, tanda baca dibuang), fullmatch salah satu pola halusinasi yang
# umum diketahui. Pakai fullmatch (bukan search) supaya HANYA menangkap
# kalau kalimatnya memang persis pola halusinasi itu - bukan kalimat
# sungguhan yang kebetulan menyinggung kata sejenis.
_HALLUCINATION_FULL_PATTERNS = [
    r"thank(s| you)( so much)? for watching"
    r"( don t forget to (like )?(share )?and subscribe)?",
    r"please (like )?(share )?and subscribe",
    r"don t forget to (like )?(share )?and subscribe",
    r"see you (in the )?next (video|time)",
    r"terima kasih (sudah |telah )?menonton",
    r"jangan lupa (like|subscribe|follow)( dan (like|subscribe|follow))*",
]


def _normalize_for_hallucination_check(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_whisper_hallucination(text: str) -> bool:
    normalized = _normalize_for_hallucination_check(text)

    if not normalized:
        return False

    for pattern in _HALLUCINATION_FULL_PATTERNS:
        if re.fullmatch(pattern, normalized):
            return True

    return False


# ============================================================
# ASR HELPER
# ============================================================

# ============================================================
# PERBAIKAN KATA YANG "NEMPEL" (PREPOSISI + NAMA TEMPAT)
# ============================================================
#
# Whisper (termasuk model fine-tuned untuk id) kadang menggabungkan
# preposisi dengan nama tempat setelahnya jadi satu kata kalau diucapkan
# tanpa jeda jelas (mis. "ke bandung" -> "kebandung"). Ini murni
# perilaku decoding model berdasarkan confidence akustik saat itu,
# BUKAN sesuatu yang bisa dikontrol langsung lewat kode - tapi karena
# pola nama tempat pada skenario demo sudah diketahui, kita bisa
# perbaiki lewat pencocokan teks sesudah ASR selesai.
#
# INI HANYA STRING MATCHING BIASA (tidak menyentuh model AI sama sekali),
# jadi risikonya jauh lebih rendah dibanding percobaan proteksi token
# sebelumnya yang gagal. Tambahkan nama tempat lain di
# _KNOWN_PLACES_FOR_SPACING_FIX sesuai skenario demo Anda.
_PREPOSITION_PLACE_FIX_PREFIXES = ["ke", "di", "dari", "menuju"]

_KNOWN_PLACES_FOR_SPACING_FIX = [
    "bandung", "jakarta", "surabaya", "yogyakarta", "semarang",
    "medan", "makassar", "palembang", "denpasar", "malang",
    "bogor", "bekasi", "bali",
]


def _fix_merged_preposition_place_names(text: str) -> str:
    for prefix in _PREPOSITION_PLACE_FIX_PREFIXES:
        for place in _KNOWN_PLACES_FOR_SPACING_FIX:
            merged = f"{prefix}{place}"
            pattern = re.compile(
                r"\b" + re.escape(merged) + r"\b",
                re.IGNORECASE,
            )
            text = pattern.sub(f"{prefix} {place}", text)
    return text


# ============================================================
# PERAPIAN RINGAN OUTPUT ASR (KHUSUS ARAH id-en)
# ============================================================
#
# Model Whisper HASIL FINE-TUNING (dipakai untuk arah id-en) cenderung
# tidak menghasilkan tanda baca sama sekali - kemungkinan besar karena
# data training fine-tuning-nya juga tidak memakai tanda baca. Model
# Inggris (base, tidak di-fine-tune) tidak mengalami ini.
#
# Perbaikan PENUH (koma, tanda tanya di tengah kalimat) butuh model
# punctuation-restoration terpisah atau retraining ulang - di luar
# scope perbaikan cepat ini. Fungsi ini HANYA kapitalisasi huruf
# pertama + titik di akhir kalau belum ada.
def _light_punctuation_cleanup(text: str) -> str:
    text = text.strip()

    if not text:
        return text

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?":
        text += "."

    return text


def _run_asr(
    signal: np.ndarray,
    direction: str,
) -> str:

    if signal is None:
        return ""

    if len(signal) == 0:
        return ""

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    whisper_lang = (
        "indonesian"
        if direction == "id-en"
        else "english"
    )

    asr_pipeline = (
        asr
        if direction == "id-en"
        else asr_en
    )

    result = asr_pipeline(
        signal,
        generate_kwargs={
            "max_new_tokens": 128,
            "language": whisper_lang,
            "task": "transcribe",
            "temperature": 0.0,
            "condition_on_prev_tokens": False,
        },
    )

    text = result.get(
        "text",
        ""
    )

    text = text.strip()

    if direction == "id-en":
        text = _fix_merged_preposition_place_names(text)
        text = _light_punctuation_cleanup(text)

    return text


# ============================================================
# NDTW
# ============================================================

def _run_ndtw_comparison(
    query_features,
    active_anchor,
    fixed_r: int
):

    n = query_features.shape[0]

    m = active_anchor.shape[0]

    r_adaptive = compute_adaptive_window(
        n,
        m,
        margin=1
    )


    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    t0 = time.perf_counter()

    alignment_adaptive = dtw_align(
        query_features,
        active_anchor,
        window_size=r_adaptive,
    )

    t_ndtw = (
        time.perf_counter() - t0
    )


    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    selisih_nm = abs(
        n - m
    )

    if fixed_r >= selisih_nm:

        t0 = time.perf_counter()

        alignment_fixed = dtw_align(
            query_features,
            active_anchor,
            window_size=fixed_r,
        )

        fixed_result = {
            "status": "berhasil",
            "r": fixed_r,
            "normalized_distance": round(
                float(
                    alignment_fixed.normalized_distance
                ),
                4
            ),
            "waktu_sec": round(
                time.perf_counter() - t0,
                4
            ),
        }

    else:

        fixed_result = {
            "status": "gagal",
            "r": fixed_r,
            "alasan": (
                f"r ({fixed_r}) < "
                f"|N-M| ({selisih_nm}), "
                "syarat kelayakan "
                "r >= |N-M| tidak terpenuhi"
            ),
        }


    comparison = {

        "adaptive": {
            "status": "berhasil",
            "r": int(r_adaptive),
            "normalized_distance": round(
                float(
                    alignment_adaptive.normalized_distance
                ),
                4
            ),
            "waktu_sec": round(
                t_ndtw,
                4
            ),
        },

        "fixed": fixed_result,

    }


    return (
        n,
        m,
        r_adaptive,
        alignment_adaptive,
        t_ndtw,
        comparison,
    )


# ============================================================
# INDEX
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# AUDIO FILE
# ============================================================

@app.route(
    "/static/audio_output/<path:filename>"
)
def audio_output(filename):

    return send_from_directory(
        AUDIO_OUTPUT_DIR,
        filename,
    )


# ============================================================
# TRANSLATE
# ============================================================

@app.route(
    "/translate",
    methods=["POST"]
)
def translate():

    request_start = time.perf_counter()


    # ========================================================
    # VALIDASI REQUEST
    # ========================================================

    if "audio" not in request.files:

        return jsonify({
            "error": "Tidak ada file audio."
        }), 400


    direction = request.form.get(
        "direction",
        "id-en"
    )

    if direction not in (
        "id-en",
        "en-id",
    ):

        return jsonify({
            "error": (
                "Direction tidak valid. "
                "Gunakan id-en atau en-id."
            )
        }), 400


    try:

        fixed_r = int(
            request.form.get(
                "fixed_r",
                100
            )
        )

    except ValueError:

        return jsonify({
            "error": "fixed_r harus berupa angka."
        }), 400


    if fixed_r < 0:

        return jsonify({
            "error": "fixed_r tidak boleh negatif."
        }), 400


    audio_file = request.files["audio"]


    if not audio_file.filename:

        return jsonify({
            "error": "Nama file audio kosong."
        }), 400


    # ========================================================
    # SAVE TEMP AUDIO
    # ========================================================

    temp_filename = (
        f"_temp_upload_{uuid.uuid4().hex}.wav"
    )

    temp_path = (
        DATA_DIR / temp_filename
    )


    try:

        audio_file.save(
            str(temp_path)
        )


        # ====================================================
        # LOAD AUDIO
        # ====================================================

        import librosa

        signal, sr = librosa.load(
            str(temp_path),
            sr=16000,
            mono=True,
        )

        signal = np.asarray(
            signal,
            dtype=np.float32
        )


        if len(signal) == 0:

            return jsonify({
                "error": "Audio kosong."
            }), 400


        audio_duration = (
            len(signal) / sr
        )


        if audio_duration < 0.3:

            return jsonify({
                "error": (
                    "Audio terlalu pendek "
                    "(kurang dari 0.3 detik)."
                )
            }), 400


        # ====================================================
        # BASIC AUDIO CHECK
        # ====================================================

        rms_original = float(
            np.sqrt(
                np.mean(
                    signal.astype(
                        np.float64
                    ) ** 2
                )
            )
        )

        peak_original = float(
            np.max(
                np.abs(signal)
            )
        )


        if (
            rms_original < 0.005
            or peak_original < 0.01
        ):

            return jsonify({
                "error": (
                    "Audio terlalu lemah. "
                    "Coba bicara lebih dekat "
                    "ke mikrofon."
                )
            }), 400


        # ====================================================
        # VAD
        # ====================================================

        t0 = time.perf_counter()


        vad_info = get_vad_info(
            signal,
            sr,
            padding_ms=VAD_PADDING_MS,
            frame_ms=25.0,
            hop_ms=10.0,
            energy_threshold_ratio=(
                VAD_ENERGY_THRESHOLD_RATIO
            ),
            min_speech_ms=(
                VAD_MIN_SPEECH_MS
            ),
            min_silence_ms=(
                VAD_MIN_SILENCE_MS
            ),
        )


        clean_signal = trim_edges_only(
            signal,
            sr,
            padding_ms=VAD_PADDING_MS,
            frame_ms=25.0,
            hop_ms=10.0,
            energy_threshold_ratio=(
                VAD_ENERGY_THRESHOLD_RATIO
            ),
            min_speech_ms=(
                VAD_MIN_SPEECH_MS
            ),
            min_silence_ms=(
                VAD_MIN_SILENCE_MS
            ),
        )


        t_vad = (
            time.perf_counter() - t0
        )


        if len(clean_signal) == 0:

            return jsonify({
                "error": (
                    "VAD tidak mendeteksi "
                    "ucapan yang cukup jelas."
                )
            }), 400


        clean_duration = (
            len(clean_signal) / sr
        )


        # ====================================================
        # ASR ORIGINAL
        #
        # Diagnostic:
        # Whisper dijalankan pada audio original
        # untuk mengetahui apakah VAD menyebabkan
        # kerusakan transkripsi.
        # ====================================================

        t0 = time.perf_counter()

        original_text = _run_asr(
            signal,
            direction,
        )

        t_asr_original = (
            time.perf_counter() - t0
        )


        # ====================================================
        # ASR AFTER VAD
        # ====================================================

        t0 = time.perf_counter()

        source_text = _run_asr(
            clean_signal,
            direction,
        )

        t_asr = (
            time.perf_counter() - t0
        )


        if _looks_like_whisper_hallucination(source_text):

            return jsonify({
                "error": (
                    "Tidak ada ucapan yang terdeteksi. "
                    "Coba rekam ulang sambil berbicara "
                    "dengan jelas."
                )
            }), 400


        # ====================================================
        # TRANSLATION
        # ====================================================

        t0 = time.perf_counter()

        target_text = _translate_text(
            source_text,
            direction,
        )

        t_mt = (
            time.perf_counter() - t0
        )


        # ====================================================
        # TTS
        # ====================================================

        t0 = time.perf_counter()

        audio_filename = _generate_tts(
            target_text,
            direction,
        )

        t_tts = (
            time.perf_counter() - t0
        )


        audio_url = (
            f"/static/audio_output/"
            f"{audio_filename}"
        )


        # ====================================================
        # WAV2VEC2
        # ====================================================

        t0 = time.perf_counter()

        query_features = (
            wav2vec_extractor.extract(
                clean_signal,
                sr,
            )
        )

        t_wav2vec = (
            time.perf_counter() - t0
        )


        # ====================================================
        # ACTIVE ANCHOR
        # ====================================================

        active_anchor = (
            acoustic_anchor_id
            if direction == "id-en"
            else acoustic_anchor_en
        )


        # ====================================================
        # NDTW
        # ====================================================

        (
            n,
            m,
            r_adaptive,
            alignment_adaptive,
            t_ndtw,
            comparison,
        ) = _run_ndtw_comparison(
            query_features,
            active_anchor,
            fixed_r,
        )


        # ====================================================
        # RTF
        # ====================================================

        rtf = (
            t_vad
            + t_asr
            + t_mt
            + t_wav2vec
            + t_ndtw
        ) / max(
            audio_duration,
            0.001
        )


        total_time = (
            time.perf_counter()
            - request_start
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        response_data = jsonify({

            # ------------------------------------------------
            # TEKS
            # ------------------------------------------------

            "id_text": (
                source_text
                if direction == "id-en"
                else target_text
            ),

            "en_text": (
                target_text
                if direction == "id-en"
                else source_text
            ),

            "direction": direction,


            # ------------------------------------------------
            # AUDIO
            # ------------------------------------------------

            "audio_url": audio_url,


            # ------------------------------------------------
            # DIAGNOSTIC ASR
            # ------------------------------------------------

            "diagnostic": {

                "asr_original": original_text,

                "asr_after_vad": source_text,

                "perbandingan": {
                    "original": original_text,
                    "after_vad": source_text,
                },

                "indikasi_vad_mengubah_asr": (
                    original_text.lower().strip()
                    != source_text.lower().strip()
                ),

                "waktu_asr_original_sec": round(
                    t_asr_original,
                    3
                ),

            },


            # ------------------------------------------------
            # VAD
            # ------------------------------------------------

            "vad": {

                "padding_ms": VAD_PADDING_MS,

                "energy_threshold_ratio": (
                    VAD_ENERGY_THRESHOLD_RATIO
                ),

                "min_speech_ms": (
                    VAD_MIN_SPEECH_MS
                ),

                "min_silence_ms": (
                    VAD_MIN_SILENCE_MS
                ),

                **vad_info,

            },


            # ------------------------------------------------
            # TECHNICAL STATS
            # ------------------------------------------------

            "technical_stats": {

                "N_frame_ujaran": int(n),

                "M_frame_referensi": int(m),

                "r_adaptive": int(
                    r_adaptive
                ),

                "normalized_distance": round(
                    float(
                        alignment_adaptive
                        .normalized_distance
                    ),
                    4
                ),

                "audio_duration_sec": round(
                    audio_duration,
                    2
                ),

                "audio_after_vad_duration_sec": round(
                    clean_duration,
                    2
                ),

                "rtf": round(
                    rtf,
                    4
                ),

                "breakdown_waktu": {

                    "vad_sec": round(
                        t_vad,
                        3
                    ),

                    "asr_whisper_original_sec": round(
                        t_asr_original,
                        3
                    ),

                    "asr_whisper_sec": round(
                        t_asr,
                        3
                    ),

                    "mt_translate_sec": round(
                        t_mt,
                        3
                    ),

                    "wav2vec2_sec": round(
                        t_wav2vec,
                        3
                    ),

                    "ndtw_only_sec": round(
                        t_ndtw,
                        3
                    ),

                    "tts_sec": round(
                        t_tts,
                        3
                    ),

                    "total_request_sec": round(
                        total_time,
                        3
                    ),

                },

            },


            # ------------------------------------------------
            # NDTW COMPARISON
            # ------------------------------------------------

            "comparison": comparison,

        })

        print()
        print("-" * 50)
        print("BREAKDOWN WAKTU PROSES (detik):")
        print(f"  VAD              : {t_vad:.3f}")
        print(f"  Wav2Vec2         : {t_wav2vec:.3f}")
        print(f"  NDTW             : {t_ndtw:.3f}")
        print(f"  ASR (Whisper)    : {t_asr:.3f}")
        print(f"  MT (Translate)   : {t_mt:.3f}")
        print(f"  TTS              : {t_tts:.3f}")
        print(f"  TOTAL            : {total_time:.3f}")
        print("-" * 50)
        print()

        return response_data


    except Exception as exc:

        print()
        print("=" * 70)
        print("ERROR PADA /translate")
        print("=" * 70)

        import traceback

        traceback.print_exc()

        print("=" * 70)
        print()

        return jsonify({
            "error": (
                "Terjadi kesalahan saat "
                "memproses audio."
            ),
            "detail": str(exc),
        }), 500


    finally:

        # ====================================================
        # HAPUS TEMP FILE
        # ====================================================

        try:

            if temp_path.exists():
                temp_path.unlink()

        except Exception as exc:

            print(
                "Peringatan: gagal menghapus "
                f"temporary file: {exc}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "Server Flask berjalan di:"
    )
    print(
        "http://127.0.0.1:5000"
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )