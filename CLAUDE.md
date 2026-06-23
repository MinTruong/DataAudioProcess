# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

# Run pipeline on a single video
python pipeline_tts.py <video_url> [--output dir] [--raw dir] [--config config.yaml]

# Run pipeline on a channel (first N videos)
python pipeline_tts.py <channel_url> --max-videos 5

# Batch process
python scripts/batch_process.py --file urls.txt

# Upload to HuggingFace (dùng config YAML)
python scripts/push_to_hub.py --config configs/upload.yaml

# Upload to HuggingFace (CLI trực tiếp)
python scripts/push_to_hub.py --dir dataset/ --repo user/dataset-name

# Run tests
python -m pytest tests/ -v

# Install dependencies
pip install -r requirements.txt

## Architecture

This project is a modular TTS dataset pipeline for Vietnamese audio books.

Pipeline flow: YouTube URL → downloader → parser → processor → exporter → Parquet + WAV

### Modules (tts_pipeline/)
- `config.py` — Pydantic settings, YAML loading, ffmpeg validation
- `downloader.py` — yt-dlp wrapper for audio + auto-caption (Vietnamese)
- `parser.py` — VTT caption parser, YouTube fragment merging
- `processor.py` — Text cleaning, overlap dedup, long segment splitting
- `exporter.py` — ffmpeg audio cutting, Parquet export, train/test split
- `cli.py` — CLI entry point (argparse)

### Output format
| Column | Type | Description |
|--------|------|-------------|
| `audio` | struct | `{path: string, bytes: binary}` — embedded audio |
| `transcription` | string | Vietnamese text |
| `file_name` | string | WAV filename (`audio/xxx.wav`) |

### Key design decisions
- YouTube auto-caption VTT is used as the transcription source (no Whisper needed)
- WAV output: mono, 22050Hz, 16-bit (standard for TTS)
- Segment duration: 2-20s, sentence-boundary splitting
- Train/test split via export (default 90/10)
- Config via YAML, overridable via CLI args

### Quality notes
- YouTube fragments (<0.3s) are merged during parsing
- Overlapping text between consecutive segments is deduplicated
- ffmpeg must be on PATH (validated at startup)
