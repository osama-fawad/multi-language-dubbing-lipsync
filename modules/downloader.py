# modules/downloader.py
import os
from pathlib import Path

def download_youtube(url: str, out_dir: Path, filename_stem: str = "input") -> Path:
    """
    Download a single YouTube video to MP4 using yt_dlp.
    Returns the full path to the saved video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_stem}.mp4"

    try:
        import yt_dlp
        ydl_opts = {
            "outtmpl": str(out_path),
            "format": "mp4/bestaudio/best",
            "quiet": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except ImportError:
        # fallback to command line if module not available
        cmd = f'yt-dlp -o "{out_path}" -f mp4/bestaudio/best "{url}"'
        print("[info] yt_dlp module missing, falling back to shell:", cmd)
        os.system(cmd)

    if not out_path.exists():
        raise FileNotFoundError("YouTube download failed.")
    print("✅ Downloaded:", out_path)
    return out_path
