# config.py
from pathlib import Path

# ---- Defaults (override via CLI flags in main.py) ----
DEFAULT_YT_URL   = ""   # e.g., "https://www.youtube.com/watch?v=xxxx"
DEFAULT_BASENAME = "monologue_clip"  # used for filenames

# Language choices for translation (NLLB codes)
LANG_NAME_TO_CODE = {
    "hindi":  "hin_Deva",
    "arabic": "arb_Arab",
    "french": "fra_Latn",
    # add more if you need
}

# Edge TTS voices (optional preference; auto-picked if None)
EDGE_LOCALE_PREFIX = {
    "hindi":  "hi-IN",
    "arabic": "ar-",
    "french": "fr-",
}
EDGE_PREFERRED_VOICE = {
    "hindi":  "hi-IN-NeerjaNeural",
    "arabic": "ar-EG-SalmaNeural",
    "french": "fr-FR-DeniseNeural",
}

# Wav2Lip checkpoint path (put correct files here)
W2L_CKPT = "Wav2Lip/checkpoints/wav2lip.pth"  # or "Wav2Lip/checkpoints/wav2lip_gan.pth"

# Output folders
DIRS = {
    "downloads":   Path("downloads"),
    "audio":       Path("audio"),
    "transcripts": Path("transcripts"),
    "translations":Path("translations"),
    "tts":         Path("tts_outputs"),
    "outputs":     Path("outputs"),
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Audio sample rates
ASR_SR   = 16000  # extract video audio to 16k for whisper
W2L_SR   = 16000  # Wav2Lip expects 16k wav typically
EDGE_SR  = 24000  # Edge TTS native synth rate; we resample to W2L_SR

# Wav2Lip runner defaults
W2L_PADS          = (0, 12, 0, 0)
W2L_RESIZE_FACTOR = 1
W2L_FORCE_FPS     = 24
