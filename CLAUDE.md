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
│  merge_cues  │  2702 raw VTT cues → 143 merged segments
│              │  - Fragments (<0.3s): extend time only, never overwrite text
│              │  - Word-level cues: combine if same utterance
│              │  - Gặp .!? bắt đầu segment mới
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  clean_text  │  Normalize whitespace
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ dedup_consecutive │  Xoá overlap text giữa segment kế
│ _text             │  Check suffix của prev ở MỌI vị trí trong curr
│                   │  (max 15 từ, word-boundary, iterate all positions)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ fix_time_overlaps │  Đẩy start time nếu 2 segment chồng lấn
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ segment_by_       │  NHÓM merged segments thành group 5-20s
│ content           │  - Mỗi merged segment là atomic unit (không split nội bộ)
│                   │  - Music/noise artifacts được strip
│                   │  - Greedy grouping: gom đến 5-20s
│                   │  - Trim 0.15s đầu + 0.3s cuối → audio-text khớp
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ fix_time_overlaps │  (2nd pass) sau khi trim
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   filter     │  Bỏ segment: rỗng, <5s, <10 ký tự
└──────┬───────┘
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
- `processor.py` — Text cleaning, `dedup_consecutive_text()`, `fix_time_overlaps()`, `segment_by_content()` (group by duration, no intra-segment split)
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
- Segment duration: **5-20s**, content-aware grouping (merged segments stay atomic)
- **No intra-segment split** — prevents audio-text mismatch from VTT overlap
- Boundary trim: 0.15s đầu + 0.3s cuối mỗi group để tránh bleed
- Train/test split via export (default 90/10)
- Config via YAML, overridable via CLI args

### segment_by_content (processor.py)

Input: 143 merged segments (sau merge_cues).

**Bước 1 — Strip music:** Mỗi segment kiểm tra `_has_music_noise()`, nếu có `[âm nhạc]`/`[nhạc]`/`[music]`/`&gt;` thì strip bằng `_strip_music_noise()`, giữ content còn lại.

**Bước 2 — Greedy grouping (5-20s target):**
- `group_dur < min_dur (5s)` → bắt buộc gộp (không orphan)
- `group_dur + seg_dur > max_dur (20s)` và `group_dur >= min_dur` → emit group, bắt đầu mới
- Còn room → gộp tiếp
- Tail group luôn được emit

**Bước 3 — Boundary trim:** 0.15s đầu + 0.3s cuối mỗi group để tránh VTT overlap artifact.

**Quan trọng: KHÔNG split intra-segment.** YouTube VTT word-level cues có overlap (cùng 1 cue chứa cuối câu trước + đầu câu sau). Character-ratio proration gây audio-text mismatch. Mỗi merged segment là atomic unit → ranh giới group = ranh giới VTT thật.

### merge_cues (parser.py)

5 cases xử lý word-level cues:
- Case 1: current text extends prev → gộp (text đầy đủ hơn)
- Case 2: prev contains current → extend time, skip
- Case 3: prev ends with .!? và current không pure overlap → new segment
- Case 4: prev ends with .!? và current là pure overlap → extend time, skip
- Case 5: no punctuation → merge vào sentence hiện tại (suffix overlap dedup)

Fragments (<0.3s): **chỉ extend time, không ghi đè text**

### dedup_consecutive_text (processor.py)

Tìm suffix dài nhất của prev segment xuất hiện ở MỌI vị trí trong curr segment (substring). Strip từ start curr qua hết overlap. Max 15 từ, word-boundary check. Nếu empty sau strip → skip segment.
