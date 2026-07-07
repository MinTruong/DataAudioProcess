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

Pipeline flow: YouTube URL → downloader → parser → punctuator → vad + aligner → exporter → Parquet + WAV

### Pipeline stages

```
YouTube URL
    │
    ▼
┌──────────────┐
│  downloader  │  yt-dlp: audio (.wav) + VTT caption (.vtt)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  parser      │  parse_vtt_full() → merge_cues()
│  merge_cues  │  2702 raw VTT cues → 217 merged segments
│              │  - Fragments (<0.3s): extend time only, never overwrite text
│              │  - Word-level cues: combine if same utterance
│              │  - Gặp .!? bắt đầu segment mới
│              │  - Force-break text >250 chars without punctuation
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│  punctuator        │  restore_punctuation() + _split_by_clause()
│                    │  Thêm .!? cho VTT không dấu câu
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│  Silero-VAD        │  get_speech_intervals() → [{start,end}]
│                    │  Speech detection từ audio waveform
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│  VAD Aligner       │  group_vad_intervals() 5-20s
│                    │  align_text_to_groups() → overlap VTT cues
└──────┬────────────┘
       │
       ▼
┌──────────────┐
│  exporter    │  ffmpeg cut → WAV + Parquet (train/test split)
└──────────────┘
```

### Modules (tts_pipeline/)
- `config.py` — Pydantic settings, YAML loading, ffmpeg validation
- `downloader.py` — yt-dlp wrapper for audio + auto-caption (Vietnamese)
- `parser.py` — VTT caption parser + `merge_cues()`: YouTube fragment merging, 5-case content-aware merge
- `punctuator.py` — Vietnamese punctuation restoration, clause splitting for raw VTT text
- `vad.py` — **Silero-VAD wrapper**: `get_speech_intervals()` → speech intervals from audio
- `vad_aligner.py` — **VAD interval grouper + text aligner**: `group_vad_intervals()`, `align_text_to_groups()`
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
- **VAD-based re-segment**: Silero-VAD replaces VTT timing completely. Text from VTT, boundaries from voice activity.
- VAD threshold: 0.3 (lower than default 0.5 for audiobook with background music)
- Segment duration: **5-20s**, greedy VAD interval grouping
- **No dedup** — text from VTT cues overlap > 0.1 with VAD group
- VTT markup (`<c>...</c>`) stripped during alignment
- Train/test split via export (default 90/10)
- Config via YAML, overridable via CLI args

### vad.py (VAD scanner)

```python
speech = get_speech_intervals(audio_path, threshold=0.3, min_speech_dur=0.3, min_silence_dur=0.3)
# Returns: [{start: 201.4, end: 201.9}, ...]
```
Uses Silero-VAD's built-in `get_speech_timestamps()`. Resamples audio to 16kHz for VAD. Returns 748 intervals for a 45-min video.

### vad_aligner.py

- `group_vad_intervals(intervals)` → greedy 5-20s groups. Emit group when adding next interval would exceed max_dur.
- `align_text_to_groups(groups, raw_cues)` → overlaps raw VTT cues with each group (ratio > 0.1), joins text, strips markup, punctuates.
- Returns 204 segments from 748 VAD intervals for a 45-min video.

### merge_cues (parser.py)

5 cases xử lý word-level cues:
- Case 1: current text extends prev → gộp (text đầy đủ hơn)
- Case 2: prev contains current → extend time, skip
- Case 3: prev ends with .!? và current không pure overlap → new segment
- Case 4: prev ends with .!? và current là pure overlap → extend time, skip
- Case 5: no punctuation → merge vào sentence hiện tại (suffix overlap dedup)

Fragments (<0.3s): **chỉ extend time, không ghi đè text**
Force-break: if len(prev_text) > 250 and no punctuation in last 150 chars → new segment.

### punctuator.py

- `restore_punctuation()`: dùng underthesea.sent_tokenize() + heuristics (từ hỏi→?, cảm thán→!, còn lại→.)
- `_split_by_clause()`: cắt text flow bằng discourse markers: "nhưng", "cho nên", "tại vì", "rồi thì", "với lại", "nói chung là"
