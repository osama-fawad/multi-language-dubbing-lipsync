# modules/wav2lip_runner.py
import os
import sys
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2


def _probe_fps(video_path: Path) -> int:
    """Read FPS from the video, with sane fallback."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps != fps or fps <= 1:  # NaN or 0/1
        return 24
    return int(round(fps))


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def _find_temp_writer_output() -> Optional[Path]:
    """
    Search for the temp file OpenCV may have produced.
    Covers both common locations and both AVI/MP4 extensions.
    """
    candidates = [
        Path("temp/result.avi"),
        Path("Wav2Lip/temp/result.avi"),
        Path("temp/result.mp4"),
        Path("Wav2Lip/temp/result.mp4"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def run_wav2lip(
    video_in: Path,
    audio_in: Path,
    checkpoint_path: Path,
    outfile: Path,
    fps: Optional[int] = None,
    pads: Tuple[int, int, int, int] = (0, 12, 0, 0),
    resize_factor: int = 1,
    box: Optional[Tuple[int, int, int, int]] = None,
    force_cpu: bool = False,
) -> Path:
    """
    Run Wav2Lip inference via the repository's inference.py, then ensure
    an MP4 exists by muxing from the writer's temp output if needed.

    Args:
        video_in:      input face video (mp4)
        audio_in:      dubbed wav (e.g., 16k mono)
        checkpoint_path: Wav2Lip checkpoint (wav2lip_gan.pth or wav2lip.pth)
        outfile:       final mp4 to write
        fps:           override FPS used by writer; if None, derived from video
        pads:          (l, t, r, b) padding for face crop
        resize_factor: 1=best quality, 2=downscale for speed/VRAM
        box:           optional (x1, y1, x2, y2) to bypass face detector
        force_cpu:     if True, disables CUDA for this call

    Returns:
        Path to the produced MP4.
    """
    video_in = Path(video_in)
    audio_in = Path(audio_in)
    checkpoint_path = Path(checkpoint_path)
    outfile = Path(outfile)

    for p in [video_in, audio_in, checkpoint_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # Derive FPS if not provided
    use_fps = fps or _probe_fps(video_in)

    # Ensure output directory exists
    if outfile.parent and not outfile.parent.exists():
        outfile.parent.mkdir(parents=True, exist_ok=True)

    # Prepare command
    cmd = [
        sys.executable, "-u", "Wav2Lip/inference.py",
        "--checkpoint_path", str(checkpoint_path),
        "--face", str(video_in),
        "--audio", str(audio_in),
        "--outfile", str(outfile),
        "--pads", str(pads[0]), str(pads[1]), str(pads[2]), str(pads[3]),
        "--resize_factor", str(resize_factor),
        "--fps", str(int(round(use_fps))),
    ]
    if box is not None:
        x1, y1, x2, y2 = box
        cmd += ["--box", str(x1), str(y1), str(x2), str(y2)]

    env = os.environ.copy()
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    print("Running:\n ", " ".join(shlex.quote(c) for c in cmd))
    ret = subprocess.run(cmd, env=env, check=False)
    print("Exit code:", ret.returncode)

    # If final MP4 exists, done
    if outfile.exists() and outfile.stat().st_size > 0:
        print("✅ Wav2Lip done:", outfile)
        return outfile

    # Otherwise, try to mux from the writer's temp file
    cand = _find_temp_writer_output()
    if cand is None:
        raise SystemExit("❌ Neither final MP4 nor temp/result.* found. Check Wav2Lip logs above.")

    if not _ffmpeg_available():
        raise SystemExit("❌ ffmpeg is not available on PATH; please install ffmpeg.")

    mux_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(cand), "-i", str(audio_in),
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(outfile),
    ]
    print("Mux cmd:\n ", " ".join(shlex.quote(c) for c in mux_cmd))
    subprocess.run(mux_cmd, check=True)

    if outfile.exists() and outfile.stat().st_size > 0:
        print("✅ Muxed:", outfile)
        return outfile

    raise SystemExit("❌ Mux failed; verify ffmpeg install and temp result file.")
