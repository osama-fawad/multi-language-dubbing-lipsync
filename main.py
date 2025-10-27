"""
TO RUN THIS SCRIPT ON LOCALLY AVAILABLE VIDEO:
python main.py --video_file "/path/to/myclip.mp4" --basename myclip --lang hindi

TO RUN THIS SCRIPT ON A YOUTUBE VIDEO:
python main.py --yt_url "https://www.youtube.com/watch?v=ERNWm9aiZQw" --basename monologue_clip --lang hindi
 
"""

# main.py
import argparse
from pathlib import Path
from config import (
    DEFAULT_YT_URL, DEFAULT_BASENAME, DIRS, LANG_NAME_TO_CODE,
    EDGE_LOCALE_PREFIX, EDGE_PREFERRED_VOICE,
    ASR_SR, W2L_SR, EDGE_SR, W2L_CKPT, W2L_PADS, W2L_RESIZE_FACTOR, W2L_FORCE_FPS
)
from modules.downloader import download_youtube
from modules.media import extract_audio_ffmpeg, probe_video_info
from modules.asr_whisper import transcribe_faster_whisper
from modules.translate_nllb import translate_segments
from modules.tts_edge import build_dubbed_timeline
from modules.wav2lip_runner import run_wav2lip

def parse_args():
    ap = argparse.ArgumentParser(
        description="End-to-end: download/choose video → ASR → translate → TTS → Wav2Lip"
    )
    ap.add_argument("--yt_url", default=DEFAULT_YT_URL, help="YouTube URL (optional if --video_file is given)")
    ap.add_argument("--video_file", default="", help="Path to a local MP4. If set, skips YouTube download.")
    ap.add_argument("--basename", default=DEFAULT_BASENAME, help="Base name for generated files")
    ap.add_argument("--lang", default="hindi", choices=list(LANG_NAME_TO_CODE.keys()))
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_asr", action="store_true")
    ap.add_argument("--skip_translate", action="store_true")
    ap.add_argument("--skip_tts", action="store_true")
    ap.add_argument("--skip_wav2lip", action="store_true")
    ap.add_argument("--w2l_ckpt", default=W2L_CKPT, help="Path to wav2lip(.pth) or wav2lip_gan(.pth)")
    return ap.parse_args()

def main():
    args = parse_args()

    # -------- pick video source --------
    # If --video_file is provided, use it and skip download.
    # Else, use YouTube flow (unless --skip_download).
    basename = args.basename
    if args.video_file:
        video_mp4 = Path(args.video_file).expanduser().resolve()
        if not video_mp4.exists():
            raise FileNotFoundError(f"Local video not found: {video_mp4}")
        # If user left default basename, derive from file name for cleanliness
        if not basename or basename == DEFAULT_BASENAME:
            basename = video_mp4.stem
        print(f"✅ Using local video: {video_mp4}")
    else:
        video_mp4 = DIRS["downloads"] / f"{basename}.mp4"
        if not args.skip_download:
            if not args.yt_url:
                print("⚠️ No --yt_url given; expecting the video to already exist at:", video_mp4)
            else:
                video_mp4 = download_youtube(args.yt_url, DIRS["downloads"], basename)
        if not video_mp4.exists():
            raise FileNotFoundError(video_mp4)
        print(f"✅ Downloaded/selected video: {video_mp4}")

    # -------- extract audio (16k mono) --------
    audio_wav = DIRS["audio"] / f"{basename}_audio.wav"
    if not args.skip_asr:
        audio_wav = extract_audio_ffmpeg(video_mp4, DIRS["audio"], sr=ASR_SR)
    if not audio_wav.exists():
        raise FileNotFoundError(audio_wav)
    print(f"✅ Extracted audio: {audio_wav}")

    # -------- ASR --------
    asr_json = DIRS["transcripts"] / f"{basename}_asr.json"
    if not args.skip_asr:
        # Use small/cpu for portability; change if you want GPU
        transcribe_faster_whisper(audio_wav, asr_json, model_size="small", device="cpu")
    if not asr_json.exists():
        raise FileNotFoundError(asr_json)
    print(f"✅ ASR saved: {asr_json}")

    # -------- Translate (NLLB) --------
    tgt_code = LANG_NAME_TO_CODE[args.lang]
    trans_json = DIRS["translations"] / f"{basename}_en_to_{args.lang}.json"
    if not args.skip_translate:
        translate_segments(asr_json, tgt_code, trans_json)
    if not trans_json.exists():
        raise FileNotFoundError(trans_json)
    print(f"✅ Translation saved: {trans_json}")

    # -------- TTS (Edge), grouped, elastic timeline → 16k wav --------
    locale = EDGE_LOCALE_PREFIX[args.lang]
    preferred = EDGE_PREFERRED_VOICE.get(args.lang)
    dub_wav = DIRS["tts"] / f"{basename}_{args.lang}_dub_16k.wav"
    if not args.skip_tts:
        import asyncio
        asyncio.run(build_dubbed_timeline(
            asr_json=asr_json, trans_json=trans_json,
            orig_audio_wav=audio_wav, out_wav=dub_wav,
            locale_prefix=locale, preferred_voice=preferred,
            sr_synth=EDGE_SR, sr_out=W2L_SR
        ))
    if not dub_wav.exists():
        raise FileNotFoundError(dub_wav)
    print(f"✅ TTS dubbed wav: {dub_wav}")

    # -------- Wav2Lip inference --------
    out_mp4 = DIRS["outputs"] / f"{basename}__{args.lang}_wav2lip.mp4"
    if not args.skip_wav2lip:
        info = probe_video_info(video_mp4)
        fps = int(round(info["fps"])) if info and "fps" in info else W2L_FORCE_FPS
        run_wav2lip(
            video_in=video_mp4, audio_in=dub_wav,
            checkpoint_path=Path(args.w2l_ckpt),
            outfile=out_mp4, fps=fps,
            pads=W2L_PADS, resize_factor=W2L_RESIZE_FACTOR,
            box=None  # let detector run; pass a box=(y1,y2,x1,x2) if you want fixed crop
        )
    if out_mp4.exists():
        print("✅ FINAL:", out_mp4.resolve())

if __name__ == "__main__":
    main()
