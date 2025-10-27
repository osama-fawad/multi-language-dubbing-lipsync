# 🎬 Multi-Language Dubbing & Lip-Sync AI

An end-to-end AI-powered pipeline for automatic video dubbing with synchronized lip movements. This project takes a video in any language, transcribes it, translates it to your target language, generates natural-sounding speech, and synchronizes the lip movements using Wav2Lip.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Features

- 🎥 **YouTube & Local Video Support** - Process videos from YouTube URLs or local files
- 🎤 **Automatic Speech Recognition (ASR)** - Transcribe audio using Faster Whisper
- 🌍 **Multi-Language Translation** - Support for Hindi, Arabic, French, and more using NLLB-200
- 🗣️ **Natural Text-to-Speech** - Generate dubbed audio with Microsoft Edge TTS
- 💋 **Lip-Sync Technology** - Synchronize lip movements using Wav2Lip
- ⚡ **Optimized Pipeline** - Smart timeline management with elastic stretching and borrowing

## 🛠️ Technologies & Tools

### Core AI Models
- **[Faster Whisper](https://github.com/guillaumekln/faster-whisper)** - Fast and accurate speech recognition
- **[NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)** (Meta) - Neural machine translation (200 languages)
- **[Edge TTS](https://github.com/rany2/edge-tts)** - Microsoft Edge's text-to-speech engine
- **[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** - Lip-sync generation model

### Supporting Libraries
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - For NLLB translation model
- **Librosa** - Audio processing and manipulation
- **FFmpeg** - Video/audio extraction and encoding
- **yt-dlp** - YouTube video downloading

## 📋 Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- CUDA-capable GPU (optional, for faster processing)
- Internet connection (for downloading models and YouTube videos)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/osama-fawad/multi-language-dubbing-lipsync.git
cd multi-language-dubbing-lipsync
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Wav2Lip Models
Download the Wav2Lip checkpoint files and place them in the `Wav2Lip/checkpoints/` directory:

- **Standard Model**: [wav2lip.pth](https://github.com/Rudrabha/Wav2Lip#getting-the-weights)
- **GAN Model** (better quality): [wav2lip_gan.pth](https://github.com/Rudrabha/Wav2Lip#getting-the-weights)

```bash
# Create checkpoints directory
mkdir -p Wav2Lip/checkpoints

# Download the models (example using wget)
wget "CHECKPOINT_URL" -O Wav2Lip/checkpoints/wav2lip.pth
```

### 4. Install FFmpeg
**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## 💡 Usage

### Basic Usage - YouTube Video
```bash
python main.py \
  --yt_url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --basename my_video \
  --lang hindi
```

### Using Local Video File
```bash
python main.py \
  --video_file "/path/to/your/video.mp4" \
  --basename my_video \
  --lang hindi
```

### Available Languages
- `hindi` - Hindi (हिन्दी)
- `arabic` - Arabic (العربية)
- `french` - French (Français)

*More languages can be added in `config.py`*

### Advanced Options

#### Skip Specific Steps
```bash
python main.py \
  --video_file video.mp4 \
  --lang hindi \
  --skip_download \
  --skip_asr \
  --skip_translate
```

#### Use GAN Model for Better Quality
```bash
python main.py \
  --video_file video.mp4 \
  --lang hindi \
  --w2l_ckpt Wav2Lip/checkpoints/wav2lip_gan.pth
```

### All Available Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--yt_url` | YouTube video URL | None |
| `--video_file` | Path to local video file | None |
| `--basename` | Base name for output files | "monologue_clip" |
| `--lang` | Target language (hindi/arabic/french) | "hindi" |
| `--skip_download` | Skip YouTube download step | False |
| `--skip_asr` | Skip speech recognition step | False |
| `--skip_translate` | Skip translation step | False |
| `--skip_tts` | Skip text-to-speech step | False |
| `--skip_wav2lip` | Skip lip-sync step | False |
| `--w2l_ckpt` | Path to Wav2Lip checkpoint | `Wav2Lip/checkpoints/wav2lip.pth` |

## 🔄 Pipeline Flow

```
┌─────────────────┐
│  Input Video    │ (YouTube URL or Local File)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Video Download  │ (yt-dlp)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Extract   │ (FFmpeg → 16kHz mono WAV)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Speech-to-Text  │ (Faster Whisper)
│  Transcription  │ → transcripts/video_asr.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Translation   │ (NLLB-200)
│  (EN → Target)  │ → translations/video_en_to_lang.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text-to-Speech  │ (Edge TTS)
│  + Timeline     │ → tts_outputs/video_lang_dub_16k.wav
│  Optimization   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Lip Sync      │ (Wav2Lip)
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Output    │ outputs/video__lang_wav2lip.mp4
└─────────────────┘
```

## 📁 Project Structure

```
multi-language-dubbing-lipsync/
├── main.py                    # Main entry point
├── config.py                  # Configuration & settings
├── modules/
│   ├── downloader.py         # YouTube video downloader
│   ├── media.py              # Audio/video extraction & probing
│   ├── asr_whisper.py        # Speech recognition (Faster Whisper)
│   ├── translate_nllb.py     # Translation (NLLB-200)
│   ├── tts_edge.py           # Text-to-speech (Edge TTS)
│   └── wav2lip_runner.py     # Lip-sync inference (Wav2Lip)
├── Wav2Lip/                  # Wav2Lip model (git submodule/clone separately)
│   └── checkpoints/          # Model weights (.pth files)
├── downloads/                # Downloaded videos
├── audio/                    # Extracted audio files
├── transcripts/              # ASR transcriptions (JSON)
├── translations/             # Translated text (JSON)
├── tts_outputs/              # Generated dubbed audio
├── outputs/                  # Final lip-synced videos
└── requirements.txt          # Python dependencies
```

## ⚙️ Configuration

Edit `config.py` to customize:

### Add More Languages
```python
LANG_NAME_TO_CODE = {
    "hindi": "hin_Deva",
    "arabic": "arb_Arab",
    "french": "fra_Latn",
    "spanish": "spa_Latn",  # Add Spanish
    "german": "deu_Latn",   # Add German
}

EDGE_LOCALE_PREFIX = {
    "spanish": "es-",
    "german": "de-",
}

EDGE_PREFERRED_VOICE = {
    "spanish": "es-ES-ElviraNeural",
    "german": "de-DE-KatjaNeural",
}
```

### Adjust Audio Settings
```python
ASR_SR = 16000    # Whisper sample rate
W2L_SR = 16000    # Wav2Lip sample rate
EDGE_SR = 24000   # Edge TTS synthesis rate
```

### Wav2Lip Parameters
```python
W2L_PADS = (0, 12, 0, 0)           # Face padding (top, bottom, left, right)
W2L_RESIZE_FACTOR = 1              # Video resize factor
W2L_FORCE_FPS = 24                 # Fallback FPS if probe fails
```

## 🎨 Key Features Explained

### 1. **Smart Timeline Management**
The TTS module uses an elastic timeline algorithm that:
- Groups segments into sentences
- Borrows time from gaps between segments
- Stretches/compresses speech to fit original timing
- Maintains natural speech rhythm

### 2. **Multilingual Support**
Powered by Meta's NLLB-200 model supporting 200+ languages with state-of-the-art translation quality.

### 3. **Natural Voice Synthesis**
Uses Microsoft Edge TTS which provides:
- High-quality neural voices
- Multiple voice options per language
- Adjustable speech rate
- No API costs

### 4. **Accurate Lip Sync**
Wav2Lip generates realistic lip movements that match the dubbed audio, creating a convincing result.

## 🐛 Troubleshooting

### Common Issues

**Issue: "FFmpeg not found"**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**Issue: "CUDA out of memory"**
```python
# In asr_whisper.py, change device to "cpu"
transcribe_faster_whisper(audio_wav, asr_json, model_size="small", device="cpu")
```

**Issue: "Wav2Lip checkpoint not found"**
```bash
# Ensure you've downloaded the checkpoint
ls Wav2Lip/checkpoints/wav2lip.pth
# If missing, download from Wav2Lip repository
```

**Issue: "YouTube download fails"**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

## 📊 Performance Notes

- **ASR (Whisper Small/CPU)**: ~1-2x real-time
- **Translation (NLLB/CPU)**: ~5-10 seconds for typical video
- **TTS (Edge)**: ~2-3x real-time
- **Wav2Lip (CPU)**: ~10-20x slower than real-time
- **Wav2Lip (GPU)**: ~1-2x real-time

*GPU acceleration is highly recommended for Wav2Lip*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Lip-sync model
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - ASR engine
- [NLLB](https://ai.meta.com/research/no-language-left-behind/) - Translation model by Meta
- [Edge TTS](https://github.com/rany2/edge-tts) - Text-to-speech wrapper

## 📧 Contact

Osama Fawad - [@osama-fawad](https://github.com/osama-fawad)

Project Link: [https://github.com/osama-fawad/multi-language-dubbing-lipsync](https://github.com/osama-fawad/multi-language-dubbing-lipsync)

---

⭐ If you find this project useful, please consider giving it a star!

