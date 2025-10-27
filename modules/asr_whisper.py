# modules/asr_whisper.py
from pathlib import Path
import json

def transcribe_faster_whisper(audio_wav: Path, out_json: Path, model_size="small", device="auto"):
    """
    Transcribe with faster-whisper; save [{start,end,text}] to JSON.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device)
    segments, _ = model.transcribe(str(audio_wav), vad_filter=True)
    out = []
    for seg in segments:
        out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("✅ ASR saved:", out_json)
    return out_json
