# modules/media.py
import subprocess, json
from pathlib import Path

def extract_audio_ffmpeg(video_path: Path, out_dir: Path, sr: int = 16000) -> Path:
    """
    Extract mono WAV audio at `sr` using ffmpeg.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"{video_path.stem}_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le",
        str(out_wav)
    ]
    subprocess.run(cmd, check=True)
    print("✅ Extracted audio:", out_wav)
    return out_wav

def probe_video_info(video_path: Path) -> dict:
    """
    Get FPS and dimensions with ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=r_frame_rate,width,height", "-of", "json", str(video_path)
    ]
    out = subprocess.check_output(cmd).decode("utf-8", "ignore")
    data = json.loads(out)
    s = data["streams"][0]
    fps_str = s["r_frame_rate"]
    w, h = s["width"], s["height"]
    num, den = (int(x) for x in fps_str.split("/"))
    fps = num / den if den else float(num)
    return {"fps": fps, "w": w, "h": h}
