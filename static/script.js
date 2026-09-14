let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let activeRecordButton = null;
let activeStatusElement = null;

const errorBox = document.getElementById("errorBox");
const audioPlayer = document.getElementById("audioPlayer");
const listenBtn = document.getElementById("listenBtn");

const setText = (id, value) => { document.getElementById(id).textContent = value ?? "-"; };

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.add("visible");
}

function clearError() { errorBox.classList.remove("visible"); }

listenBtn.addEventListener("click", () => audioPlayer.play());

function setRecordingState(button, recording) {
    button.classList.toggle("recording", recording);
    button.setAttribute("aria-label", recording ? "Berhenti merekam" : button.dataset.originalLabel || "Mulai rekam");
    button.setAttribute("title", recording ? "Berhenti merekam" : button.dataset.originalLabel || "Mulai rekam");
}

async function startRecording(button, direction, statusElement) {
    clearError();
    if (isRecording) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        activeRecordButton = button;
        activeStatusElement = statusElement;
        button.dataset.recordingDirection = direction;

        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = () => {
            stream.getTracks().forEach((track) => track.stop());
            sendAudioToServer(direction, statusElement, button);
        };

        mediaRecorder.start();
        isRecording = true;
        setRecordingState(button, true);
        statusElement.textContent = "Merekam...";
    } catch (error) {
        showError("Mikrofon tidak dapat digunakan. Pastikan izin mikrofon sudah diberikan.");
    }
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    mediaRecorder.stop();
    isRecording = false;
    setRecordingState(activeRecordButton, false);
    activeRecordButton.disabled = true;
    activeStatusElement.textContent = "Memproses...";
}

const conversationRecordBtn = document.getElementById("conversationRecordBtn");
const conversationStatus = document.getElementById("conversationStatus");
const conversationSpeakerLabel = document.getElementById("conversationSpeakerLabel");
const conversationSpeakerHint = document.getElementById("conversationSpeakerHint");

conversationRecordBtn.dataset.originalLabel = conversationRecordBtn.getAttribute("aria-label");
conversationRecordBtn.addEventListener("click", () => {
    if (isRecording) {
        if (activeRecordButton === conversationRecordBtn) stopRecording();
        return;
    }
    startRecording(conversationRecordBtn, conversationRecordBtn.dataset.direction, conversationStatus);
});

document.querySelectorAll(".conversation-language").forEach((button) => {
    button.addEventListener("click", () => {
        if (isRecording) return;
        setConversationLanguage(button.dataset.direction);
    });
});

function setConversationLanguage(direction) {
    const isIdToEn = direction === "id-en";
    conversationRecordBtn.dataset.direction = direction;
    conversationRecordBtn.dataset.originalLabel = isIdToEn ? "Rekam Bahasa Indonesia" : "Record English";
    conversationRecordBtn.setAttribute("aria-label", conversationRecordBtn.dataset.originalLabel);
    conversationRecordBtn.setAttribute("title", conversationRecordBtn.dataset.originalLabel);
    conversationSpeakerLabel.textContent = isIdToEn ? "Bahasa Indonesia" : "English";
    conversationSpeakerHint.textContent = isIdToEn ? "Tekan mikrofon untuk berbicara" : "Press the microphone to speak";
    document.querySelectorAll(".conversation-language").forEach((button) => button.classList.toggle("active", button.dataset.direction === direction));
}

async function sendAudioToServer(direction, statusElement, button) {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    formData.append("direction", direction);

    try {
        const response = await fetch("/translate", { method: "POST", body: formData });
        const data = await response.json();

        if (!response.ok || data.error) throw new Error(data.error || "Server gagal memproses audio.");

        const sourceText = direction === "id-en" ? data.id_text : data.en_text;
        const targetText = direction === "id-en" ? data.en_text : data.id_text;

        audioPlayer.src = data.audio_url;
        listenBtn.classList.add("visible");

        const s = data.technical_stats;
        const b = s.breakdown_waktu;
        setText("distanceValue", s.normalized_distance);
        setText("adaptiveValue", s.r_adaptive);
        setText("rtfValue", s.rtf);
        setText("durationValue", `${s.audio_duration_sec}s`);
        setText("nValue", s.N_frame_ujaran);
        setText("mValue", s.M_frame_referensi);
        setText("fixedValue", data.comparison.fixed.status === "berhasil" ? `${data.comparison.fixed.normalized_distance} (berhasil)` : "Gagal memenuhi syarat");
        setText("vadValue", `${b.vad_sec}s`);
        setText("asrValue", `${b.asr_whisper_sec}s`);
        setText("mtValue", `${b.mt_translate_sec}s`);
        setText("ndtwValue", `${b.ndtw_only_sec}s`);
        setText("scoreState", "Pengukuran terakhir selesai");
        statusElement.textContent = "Selesai.";

        appendConversationTurn(direction, sourceText, targetText);
        setConversationLanguage(direction === "id-en" ? "en-id" : "id-en");
    } catch (err) {
        showError(err.message);
        statusElement.textContent = "Pemrosesan gagal.";
    } finally {
        button.disabled = false;
        activeRecordButton = null;
        activeStatusElement = null;
    }
}

function appendConversationTurn(direction, sourceText, targetText) {
    const log = document.getElementById("conversationLog");
    const empty = log.querySelector(".conversation-empty");
    if (empty) empty.remove();
    const item = document.createElement("article");
    item.className = "conversation-item";
    const language = direction === "id-en" ? "Bahasa Indonesia" : "English";
    const speaker = document.createElement("strong");
    speaker.textContent = language;
    const source = document.createElement("p");
    source.textContent = sourceText || "Tidak ada ucapan terdeteksi";
    const target = document.createElement("p");
    target.textContent = targetText || "Terjemahan kosong";
    item.append(speaker, source, target);
    log.appendChild(item);
}