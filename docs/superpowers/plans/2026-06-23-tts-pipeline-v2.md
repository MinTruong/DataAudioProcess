# Pipeline TTS Dataset v2 — Refactor & Enhance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor monolithic `pipeline_tts.py` into a modular, testable, configurable pipeline with quality improvements, train/test split, multi-video batching, and HuggingFace upload capability.

**Architecture:** Split into 6 independent modules (config, downloader, parser, processor, exporter, cli) + optional huggingface uploader. Each module has a single responsibility and can be tested in isolation.

**Tech Stack:** Python 3.12+, yt-dlp, pandas, pyarrow, tqdm, ffmpeg, pydantic~=2.0, huggingface-hub, pyyaml~=6.0

## Global Constraints

- Python 3.12+ required (f-string syntax, `sys.stdout.reconfigure`)
- Windows 11 compatibility (encoding fix, path handling with PureWindowsPath)
- ffmpeg must be available on PATH (checked at config load time)
- Output Parquet v2 format via pyarrow
- All text handling must use UTF-8 encoding
- Dataset format columns: `audio`, `audioduration (s)`, `transcription`, `file_name`
- Existing `pipeline_tts.py` is the baseline — new modules must match its output format exactly
- No new top-level dependencies beyond what's listed in Tech Stack

---

## File Structure

```
AudioProcess/
├── pipeline_tts.py            ← Entry point (imports from tts_pipeline/)
├── tts_pipeline/              ← NEW package
│   ├── __init__.py
│   ├── config.py              ← Pydantic config + YAML loading
│   ├── downloader.py          ← yt-dlp wrapper
│   ├── parser.py              ← VTT parsing + cue merging
│   ├── processor.py           ← dedup, split, clean, VAD (future)
│   └── exporter.py            ← cut audio + Parquet export
├── scripts/                   ← NEW
│   ├── batch_process.py       ← Multi-video batch runner
│   └── push_to_hub.py         ← HuggingFace upload helper
├── configs/                   ← NEW (user-created config files)
│   └── default.yaml
├── tests/                     ← NEW
│   ├── test_parser.py
│   ├── test_processor.py
│   └── fixtures/
│       └── sample.vtt         ← VTT fixture for tests
├── dataset/                   ← Output (unchanged)
├── raw/                       ← Raw downloads (unchanged)
├── sample_dataset/            ← Sample (unchanged)
├── SPEC.md
├── CLAUDE.md
└── requirements.txt           ← NEW
```

---

### Task 1: Project scaffolding — package structure, config, requirements

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\requirements.txt`
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\__init__.py`
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\config.py`
- Create: `D:\MinhTh_code\AudioProcess\configs\default.yaml`
- Modify: `D:\MinhTh_code\AudioProcess\pipeline_tts.py` (add `import` guard)

**Interfaces:**
- Consumes: — (first task)
- Produces: `tts_pipeline.config.Settings` — pydantic BaseSettings with all pipeline parameters; `configs/default.yaml` — default YAML

- [ ] **Step 1: Create requirements.txt**

Target file: `D:\MinhTh_code\AudioProcess\requirements.txt`

```
yt-dlp>=2024.0
pandas>=2.0
pyarrow>=14.0
tqdm>=4.60
pydantic>=2.0,<3.0
pyyaml>=6.0
huggingface-hub>=0.20
```

- [ ] **Step 2: Create package `__init__.py`**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\__init__.py`

```python
"""TTS Dataset Pipeline — YouTube audio books to TTS training datasets."""

__version__ = "2.0.0"
```

- [ ] **Step 3: Create config module**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\config.py`

The `Settings` class must cover every tunable constant currently in `pipeline_tts.py` (lines 30–41), plus new fields: `sample_rate` (22050), `min_segment_dur` (2.0), `max_segment_dur` (20.0), `min_text_len` (10), `max_text_len` (500), `merge_gap` (0.5), `fragment_threshold` (0.3), `train_split` (0.9), `ffmpeg_path` ("ffmpeg"), `youtube_caption_langs` (["vi"]), `output_dir` ("dataset"), `raw_dir` ("raw").

```python
from pathlib import Path
from typing import Literal
import subprocess
import sys

import yaml
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    # Paths
    raw_dir: Path = Field(default=Path("raw"), description="Directory for raw downloads")
    output_dir: Path = Field(default=Path("dataset"), description="Directory for final dataset")

    # Audio parameters
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    min_segment_dur: float = Field(default=2.0, ge=1.0, le=10.0)
    max_segment_dur: float = Field(default=20.0, ge=5.0, le=60.0)

    # Text filtering
    min_text_len: int = Field(default=10, ge=1, le=100)
    max_text_len: int = Field(default=500, ge=50, le=2000)

    # VTT parsing
    merge_gap: float = Field(default=0.5, ge=0.1, le=2.0, description="Max gap (s) to merge adjacent cues")
    fragment_threshold: float = Field(default=0.3, ge=0.01, le=1.0, description="Cues shorter than this are fragments")

    # Train/test split
    train_split: float = Field(default=0.9, ge=0.5, le=1.0)

    # YouTube
    youtube_caption_langs: list[str] = Field(default=["vi"])
    max_videos_per_channel: int = Field(default=5, ge=1, le=100)

    # System
    ffmpeg_path: str = Field(default="ffmpeg")
    device: Literal["cpu", "cuda"] = "cpu"

    @field_validator("ffmpeg_path")
    @classmethod
    def ffmpeg_must_exist(cls, v: str) -> str:
        try:
            subprocess.run([v, "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise ValueError(f"ffmpeg not found at '{v}'. Install ffmpeg and ensure it's on PATH.")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file, merging with defaults."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    @classmethod
    def from_cli_overrides(cls, **overrides) -> "Settings":
        """Create settings with specific overrides from CLI args."""
        return cls(**{k: v for k, v in overrides.items() if v is not None})


# Windows encoding fix (moved from main module)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
```

- [ ] **Step 4: Create default YAML config**

Target file: `D:\MinhTh_code\AudioProcess\configs\default.yaml`

```yaml
# TTS Pipeline default configuration
raw_dir: raw
output_dir: dataset
sample_rate: 22050
min_segment_dur: 2.0
max_segment_dur: 20.0
min_text_len: 10
max_text_len: 500
merge_gap: 0.5
fragment_threshold: 0.3
train_split: 0.9
youtube_caption_langs:
  - vi
max_videos_per_channel: 5
ffmpeg_path: ffmpeg
```

- [ ] **Step 5: Add import guard to existing pipeline_tts.py**

Insert at line 1 of `pipeline_tts.py`:

```python
"""
THIS FILE IS THE ENTRY POINT for the TTS pipeline.
It delegates to tts_pipeline modules.
"""
from tts_pipeline.config import Settings
from tts_pipeline.downloader import download_video_and_subs
from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.processor import clean_text, dedup_consecutive_text, split_long_segments
from tts_pipeline.exporter import export_dataset
```

Leave all function bodies in place for now — they will be moved to their respective modules in the next tasks.

- [ ] **Step 6: Install dependencies and verify**

```bash
cd D:/MinhTh_code/AudioProcess
pip install -r requirements.txt
python -c "from tts_pipeline.config import Settings; s = Settings(); print(s.model_dump())"
```

Expected output: all default settings printed as dict.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding — package structure, config, requirements"
```

---

### Task 2: Extract downloader module

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\downloader.py`
- Modify: `D:\MinhTh_code\AudioProcess\pipeline_tts.py` (replace function body with import)

**Interfaces:**
- Consumes: `tts_pipeline.config.Settings` (read `raw_dir`, `sample_rate`, `youtube_caption_langs`)
- Produces: `def download_video_and_subs(video_url: str, output_dir: str | Path, settings: Settings | None = None) -> dict` — same return dict as current: `{video_id, title, audio_path, vtt_path}`
- Produces: `def get_channel_videos(channel_url: str, max_videos: int = 5) -> list[dict]` — same as current

- [ ] **Step 1: Write the downloader module**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\downloader.py`

```python
"""YouTube video and caption download using yt-dlp."""

import subprocess
from pathlib import Path

import yt_dlp

from tts_pipeline.config import Settings


def download_video_and_subs(
    video_url: str,
    output_dir: str | Path,
    settings: Settings | None = None,
) -> dict:
    """Download audio (WAV) and auto-caption from YouTube.

    Returns:
        dict with keys: video_id, title, audio_path, vtt_path
    """
    s = settings or Settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "writesubtitles": False,
        "writeautomaticsub": True,
        "subtitleslangs": s.youtube_caption_langs,
        "subtitlesformat": "vtt",
        "skip_download": False,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info["id"]
        title = info.get("title", "unknown")

    wav_path = _resolve_wav(output_dir, video_id, s.sample_rate)
    vtt_path = _resolve_vtt(output_dir, video_id)

    return {
        "video_id": video_id,
        "title": title,
        "audio_path": str(wav_path),
        "vtt_path": str(vtt_path),
    }


def _resolve_wav(output_dir: Path, video_id: str, sample_rate: int) -> Path:
    """Find or convert audio to WAV."""
    wav_path = output_dir / f"{video_id}.wav"
    if wav_path.exists():
        return wav_path
    for f in output_dir.glob(f"{video_id}.*"):
        if f.suffix in (".wav", ".mp3", ".m4a", ".opus"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(f), "-ar", str(sample_rate), "-ac", "1", str(wav_path)],
                check=True, capture_output=True,
            )
            if f.suffix != ".wav":
                f.unlink()
            return wav_path
    raise FileNotFoundError(f"No audio file found for {video_id}")


def _resolve_vtt(output_dir: Path, video_id: str) -> Path:
    """Find the VTT caption file."""
    for p in output_dir.glob(f"{video_id}*.vtt"):
        content = p.read_text(encoding="utf-8", errors="replace")
        if "WEBVTT" in content:
            return p
    raise FileNotFoundError(f"No VTT caption found for {video_id}")


def get_channel_videos(channel_url: str, max_videos: int = 5) -> list[dict]:
    """List recent videos from a channel or playlist."""
    opts = {"quiet": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    videos = []
    if "entries" in info:
        for entry in info["entries"][:max_videos]:
            if entry:
                videos.append({
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    "title": entry.get("title", "unknown"),
                })
    return videos
```

- [ ] **Step 2: Replace function in pipeline_tts.py**

In `pipeline_tts.py`, replace the entire `download_video_and_subs` function (lines 47-106) and `get_channel_videos` function (lines 536-555) with these imports at top:

```python
from tts_pipeline.downloader import download_video_and_subs, get_channel_videos
```

Remove the original function bodies.

- [ ] **Step 3: Quick smoke test**

```bash
cd D:/MinhTh_code/AudioProcess
python -c "
from tts_pipeline.downloader import download_video_and_subs
info = download_video_and_subs('https://www.youtube.com/watch?v=iUGFXuxHNAA', 'raw')
print(info['video_id'], info['title'][:30])
"
```

Expected: prints video ID and title.

- [ ] **Step 4: Commit**

```bash
git add tts_pipeline/downloader.py pipeline_tts.py
git commit -m "refactor: extract downloader module"
```

---

### Task 3: Extract parser module (VTT parsing + cue merging)

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\parser.py`
- Modify: `D:\MinhTh_code\AudioProcess\pipeline_tts.py` (replace functions with imports)

**Interfaces:**
- Consumes: VTT file path → raw cues
- Produces: `def parse_vtt_full(vtt_path: str | Path) -> list[dict]` — returns `[{start, end, text}]`
- Produces: `def merge_cues(cues: list[dict], settings: Settings | None = None) -> list[dict]` — merged segments

- [ ] **Step 1: Write the parser module**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\parser.py`

```python
"""VTT caption parsing and cue merging for YouTube auto-captions."""

import re
from pathlib import Path

from tts_pipeline.config import Settings


def parse_vtt_time(ts_str: str) -> float:
    """Convert VTT timestamp (00:00:01.560) to seconds."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _strip_vtt_markup(text: str) -> str:
    """Remove VTT word-level timestamps and markup tags."""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt_full(vtt_path: str | Path) -> list[dict]:
    """Parse YouTube VTT caption into raw time-aligned cues.

    Returns:
        list of {start: float, end: float, text: str}
    """
    raw_cues: list[dict] = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_start: float | None = None
    current_end: float | None = None
    cue_text: list[str] = []
    in_cue = False

    for line in lines:
        line = line.rstrip("\n")
        if " --> " in line:
            if current_start is not None and cue_text:
                raw_cues.append({
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(cue_text).strip(),
                })
            parts = line.split(" --> ")
            current_start = parse_vtt_time(parts[0])
            current_end = parse_vtt_time(parts[1].split()[0])
            cue_text = []
            in_cue = True
        elif in_cue:
            clean = line.strip()
            if clean and not clean.startswith("WEBVTT") and not clean.startswith("Kind:") and not clean.startswith("Language:"):
                cue_text.append(clean)

    if current_start is not None and cue_text:
        raw_cues.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(cue_text).strip(),
        })
    return raw_cues


def merge_cues(cues: list[dict], settings: Settings | None = None) -> list[dict]:
    """Merge YouTube auto-caption fragments into complete segments.

    YouTube generates 3 cues per utterance: word-level (long), plain fragment
    (0.01s), and the next utterance start. This merges them into clean segments.

    Args:
        cues: Raw cues from parse_vtt_full()
        settings: Pipeline Settings (controls merge_gap, fragment_threshold)

    Returns:
        list of {start: float, end: float, text: str} with markup stripped
    """
    s = settings or Settings()
    if not cues:
        return []

    for cue in cues:
        cue["text"] = _strip_vtt_markup(cue["text"])

    merged: list[dict] = []
    i = 0

    while i < len(cues):
        cue = cues[i]
        dur = cue["end"] - cue["start"]

        # Fragment (< threshold): merge time into previous segment
        if dur < s.fragment_threshold and merged:
            merged[-1]["end"] = max(merged[-1]["end"], cue["end"])
            if cue["text"] and len(cue["text"]) > len(merged[-1]["text"]):
                merged[-1]["text"] = cue["text"]
            i += 1
            continue

        combined_text = cue["text"]
        combined_end = cue["end"]
        j = i + 1

        while j < len(cues):
            next_dur = cues[j]["end"] - cues[j]["start"]
            next_text = cues[j]["text"]

            # Skip fragments
            if next_dur < s.fragment_threshold:
                if cues[j]["end"] > combined_end:
                    combined_end = cues[j]["end"]
                if next_text and len(next_text) > len(combined_text):
                    combined_text = next_text
                j += 1
                continue

            gap = cues[j]["start"] - combined_end
            if gap > s.merge_gap:
                break

            if next_text.startswith(combined_text) and len(next_text) > len(combined_text):
                combined_text = next_text
                combined_end = cues[j]["end"]
                j += 1
            elif next_text == combined_text:
                combined_end = cues[j]["end"]
                j += 1
            else:
                break

        merged.append({"start": cue["start"], "end": combined_end, "text": combined_text})
        i = j

    return merged
```

- [ ] **Step 2: Replace functions in pipeline_tts.py**

Replace lines 109-290 (the entire "BUOC 2" section) with imports and maintain `clean_text` in the processor module (next task). In `pipeline_tts.py` add:

```python
from tts_pipeline.parser import parse_vtt_full, merge_cues
```

Remove the original `parse_vtt_time`, `strip_vtt_markup`, `parse_vtt_full`, `merge_cues` functions.

- [ ] **Step 3: Write a parser test**

Target file: `D:\MinhTh_code\AudioProcess\tests\test_parser.py`

```python
"""Tests for VTT parser."""

import json
from pathlib import Path

import pytest

from tts_pipeline.parser import parse_vtt_full, merge_cues, parse_vtt_time


def test_parse_vtt_time_seconds_only():
    assert parse_vtt_time("00:01.560") == pytest.approx(1.56, abs=0.01)


def test_parse_vtt_time_hms():
    assert parse_vtt_time("00:00:03.350") == pytest.approx(3.35, abs=0.01)


def test_parse_vtt_time_long():
    assert parse_vtt_time("01:02:03.456") == pytest.approx(3723.456, abs=0.01)


def test_merge_cues_empty():
    assert merge_cues([]) == []
```

- [ ] **Step 4: Run tests**

```bash
cd D:/MinhTh_code/AudioProcess
pip install pytest
python -m pytest tests/test_parser.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tts_pipeline/parser.py tests/test_parser.py pipeline_tts.py
git commit -m "refactor: extract parser module with tests"
```

---

### Task 4: Extract processor module (dedup, split, clean)

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\processor.py`
- Modify: `D:\MinhTh_code\AudioProcess\pipeline_tts.py`

**Interfaces:**
- Consumes: merged segments from parser
- Produces: `def clean_text(text: str) -> str` — whitespace normalization
- Produces: `def dedup_consecutive_text(segments: list[dict]) -> list[dict]` — overlap removal
- Produces: `def split_long_segments(segments: list[dict], max_dur: float = 20.0) -> list[dict]` — long segment splitting

- [ ] **Step 1: Write processor module**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\processor.py`

```python
"""Text cleaning, deduplication, and segment splitting."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def dedup_consecutive_text(segments: list[dict]) -> list[dict]:
    """Remove overlapping text between consecutive segments.

    YouTube captions cause the last 1-3 words of segment N to appear
    at the start of segment N+1. This strips the overlap.
    """
    result: list[dict] = []
    for seg in segments:
        text = seg["text"]

        if result:
            prev_text = result[-1]["text"]
            prev_words = prev_text.split()

            # Check 3-word overlap
            if len(prev_words) >= 3:
                candidate = " ".join(prev_words[-3:])
                if text.startswith(candidate):
                    text = text[len(candidate):].strip().lstrip(",").strip()
                    if not text:
                        continue
                    text = text[0].upper() + text[1:] if text else text

            # Check 2-word overlap
            if text and result and len(prev_words) >= 2:
                candidate = " ".join(prev_words[-2:])
                if text.startswith(candidate):
                    text = text[len(candidate):].strip().lstrip(",").strip()
                    if not text:
                        continue
                    text = text[0].upper() + text[1:] if text else text

            # Check 1-word overlap
            if text and result:
                last_word = prev_words[-1] if prev_words else ""
                if last_word and len(last_word) > 1 and text.startswith(last_word):
                    text = text[len(last_word):].strip().lstrip(",").strip()
                    if not text:
                        continue
                    text = text[0].upper() + text[1:] if text else text

        result.append({**seg, "text": text})

    return result


def split_long_segments(
    segments: list[dict], max_dur: float = 20.0, min_dur: float = 2.0
) -> list[dict]:
    """Split segments longer than max_dur by sentence boundary.

    Time is prorated by character count across sentences.
    """
    result: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg["text"]

        if dur <= max_dur:
            result.append(seg)
            continue

        # Try sentence split, fallback to comma split
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) < 2:
            parts = re.split(r"(?<=[,])\s+", text)
        if len(parts) < 2:
            result.append(seg)
            continue

        total_chars = sum(len(s) for s in parts)
        if total_chars == 0:
            result.append(seg)
            continue

        char_ratio = dur / total_chars
        current_start = seg["start"]

        for sent in parts:
            sent = sent.strip()
            if not sent:
                continue
            sent_dur = max(len(sent) * char_ratio, min_dur)
            result.append({
                "start": current_start,
                "end": current_start + sent_dur,
                "text": sent,
            })
            current_start += sent_dur

    return result
```

- [ ] **Step 2: Replace functions in pipeline_tts.py**

Add import:
```python
from tts_pipeline.processor import clean_text, dedup_consecutive_text, split_long_segments
```

Remove original `clean_text`, `dedup_consecutive_text`, `split_long_segments` function bodies.

- [ ] **Step 3: Write processor tests**

Target file: `D:\MinhTh_code\AudioProcess\tests\test_processor.py`

```python
"""Tests for text processor."""

from tts_pipeline.processor import clean_text, dedup_consecutive_text, split_long_segments


def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_strips_edges():
    assert clean_text("\n\nfoo\n\n") == "foo"


def test_dedup_consecutive_no_change():
    segs = [{"start": 0, "end": 2, "text": "Hello world."}]
    result = dedup_consecutive_text(segs)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world."


def test_dedup_removes_three_word_overlap():
    segs = [
        {"start": 0, "end": 5, "text": "Một hai ba bốn năm"},
        {"start": 5, "end": 10, "text": "bốn năm sáu bảy tám"},
    ]
    result = dedup_consecutive_text(segs)
    assert len(result) == 2
    assert result[1]["text"] == "sáu bảy tám"


def test_split_long_segments_short_unchanged():
    segs = [{"start": 0, "end": 5, "text": "Short text."}]
    result = split_long_segments(segs, max_dur=20.0)
    assert len(result) == 1


def test_split_long_segments_splits():
    segs = [{"start": 0, "end": 25, "text": "Câu thứ nhất. Câu thứ hai. Câu thứ ba."}]
    result = split_long_segments(segs, max_dur=20.0)
    assert len(result) >= 2
```

- [ ] **Step 4: Run tests**

```bash
cd D:/MinhTh_code/AudioProcess
python -m pytest tests/ -v
```

Expected: ~10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tts_pipeline/processor.py tests/test_processor.py pipeline_tts.py
git commit -m "refactor: extract processor module with tests"
```

---

### Task 5: Extract exporter module (cut audio + Parquet)

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\exporter.py`
- Modify: `D:\MinhTh_code\AudioProcess\pipeline_tts.py`

**Interfaces:**
- Consumes: merged+processed segments, audio source path, settings
- Produces: `def cut_audio_segment(audio_path: str, output_path: str, start: float, end: float, sample_rate: int = 22050) -> tuple[bool, str]`
- Produces: `def export_dataset(segments: list[dict], output_dir: str | Path, video_id: str, audio_source_path: str, settings: Settings | None = None, split_train_test: bool = True) -> dict`

- [ ] **Step 1: Write exporter module**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\exporter.py`

```python
"""Audio cutting and Parquet dataset export."""

import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tts_pipeline.config import Settings
from tts_pipeline.processor import clean_text


def cut_audio_segment(
    audio_path: str,
    output_path: str,
    start: float,
    end: float,
    sample_rate: int = 22050,
) -> tuple[bool, str]:
    """Cut one audio segment with ffmpeg.

    Returns:
        (success: bool, message_or_path: str)
    """
    if end - start < 2.0 or end - start > 20.0:
        return False, f"Duration {end - start:.1f}s out of range"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(end - start),
        "-i", str(audio_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, str(output_path)
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode("utf-8", errors="replace")


def _export_split(
    df: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    video_id: str,
) -> Path:
    """Save one train/test split to Parquet."""
    parquet_path = output_dir / f"{video_id}_{split_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def export_dataset(
    segments: list[dict],
    output_dir: str | Path,
    video_id: str,
    audio_source_path: str,
    settings: Settings | None = None,
    split_train_test: bool = True,
) -> dict:
    """Cut audio segments and export dataset as Parquet.

    Args:
        segments: List of {start, end, text} dicts.
        output_dir: Root output directory.
        video_id: YouTube video ID (used for naming).
        audio_source_path: Path to the full audio WAV file.
        settings: Pipeline settings (controls sample_rate, train_split).
        split_train_test: If True, split into train/test Parquet files.

    Returns:
        dict with {video_id, total_segments, success, fail, parquet_paths: str|None}
    """
    s = settings or Settings()
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    success = 0
    fail = 0

    for idx, seg in enumerate(tqdm(segments, desc=f"Cut audio [{video_id}]")):
        dur = seg.get("duration", seg["end"] - seg["start"])
        text = clean_text(seg["text"])

        if dur < s.min_segment_dur or dur > s.max_segment_dur:
            fail += 1
            continue
        if not text or len(text) < s.min_text_len or len(text) > s.max_text_len:
            fail += 1
            continue

        audio_filename = f"{video_id}_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename

        ok, _ = cut_audio_segment(
            audio_source_path, str(audio_path),
            seg["start"], seg["end"], s.sample_rate,
        )
        if ok:
            rows.append({
                "audio": str(audio_path.relative_to(output_dir)),
                "audioduration (s)": round(dur, 2),
                "transcription": text,
                "file_name": audio_filename,
            })
            success += 1
        else:
            fail += 1

    result = {
        "video_id": video_id,
        "total_segments": len(segments),
        "success": success,
        "fail": fail,
        "parquet_paths": None,
    }

    if not rows:
        print(f"\n[!] No valid segments for {video_id}")
        return result

    df = pd.DataFrame(rows)

    if split_train_test and s.train_split < 1.0:
        n_train = int(len(df) * s.train_split)
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]
        train_path = _export_split(train_df, "train", output_dir, video_id)
        test_path = _export_split(test_df, "test", output_dir, video_id)
        result["parquet_paths"] = {"train": str(train_path), "test": str(test_path)}
        print(f"\n[OK] Train: {len(train_df)}, Test: {len(test_df)}")
    else:
        path = _export_split(df, "dataset", output_dir, video_id)
        result["parquet_paths"] = str(path)
        print(f"\n[OK] Dataset: {path} ({len(df)} samples)")

    return result
```

- [ ] **Step 2: Replace functions in pipeline_tts.py**

Add import:
```python
from tts_pipeline.exporter import cut_audio_segment, export_dataset
```

Remove original `cut_audio_segment` and `export_dataset` function bodies.

- [ ] **Step 3: Run smoke test**

```bash
cd D:/MinhTh_code/AudioProcess
rm -rf dataset/audio dataset/*.parquet raw/*
python pipeline_tts.py "https://www.youtube.com/watch?v=iUGFXuxHNAA" 2>&1 | tail -20
```

Expected: pipeline runs and produces dataset.

- [ ] **Step 4: Commit**

```bash
git add tts_pipeline/exporter.py pipeline_tts.py
git commit -m "refactor: extract exporter module with train/test split"
```

---

### Task 6: Refactor main pipeline into CLI module

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\tts_pipeline\cli.py`
- Modify: `D:\MinhTh_code\AudioProcess\pipeline_tts.py` (replace `if __name__` block with call to `cli.main`)

**Interfaces:**
- Consumes: all previous modules
- Produces: `def main(argv: list[str] | None = None) -> None` — CLI entry point
- Produces: `def run_pipeline(video_url: str, raw_dir: str | None, output_dir: str | None, settings: Settings | None = None) -> dict`

- [ ] **Step 1: Write CLI module**

Target file: `D:\MinhTh_code\AudioProcess\tts_pipeline\cli.py`

```python
"""CLI entry point for the TTS pipeline."""

import argparse
import sys

from tts_pipeline.config import Settings
from tts_pipeline.downloader import download_video_and_subs, get_channel_videos
from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.processor import clean_text, dedup_consecutive_text, split_long_segments
from tts_pipeline.exporter import export_dataset


def run_pipeline(
    video_url: str,
    raw_dir: str | None = None,
    output_dir: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run the full pipeline for a single video URL."""
    s = settings or Settings()
    raw = raw_dir or str(s.raw_dir)
    out = output_dir or str(s.output_dir)

    print(f"\n{'='*60}")
    print(f"[PROCESS] {video_url}")
    print(f"{'='*60}")

    # Step 1: Download
    print("\n[Step 1] Download video + caption...")
    video_info = download_video_and_subs(video_url, raw, s)
    print(f"   ID: {video_info['video_id']}")
    print(f"   Title: {video_info['title']}")

    # Step 2: Parse + merge
    print("\n[Step 2] Parse caption & merge cues...")
    raw_cues = parse_vtt_full(video_info["vtt_path"])
    print(f"   Raw cues: {len(raw_cues)}")
    merged = merge_cues(raw_cues, s)
    print(f"   After merge: {len(merged)}")

    for seg in merged:
        seg["text"] = clean_text(seg["text"])

    merged = dedup_consecutive_text(merged)
    merged = [seg for seg in merged if seg["text"] and len(seg["text"]) >= s.min_text_len]
    print(f"   After dedup: {len(merged)}")

    for seg in merged[:3]:
        dur = seg["end"] - seg["start"]
        print(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s] ({dur:.1f}s) {seg['text'][:60]}")

    merged = split_long_segments(merged, s.max_segment_dur, s.min_segment_dur)
    print(f"   After split: {len(merged)}")

    # Step 3+4: Export
    print("\n[Step 3+4] Cut audio & export dataset...")
    result = export_dataset(
        merged, out, video_info["video_id"],
        video_info["audio_path"], s,
        split_train_test=True,
    )

    print(f"\n[OK] Raw: {raw}")
    print(f"[OK] Dataset: {out}")
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TTS Dataset from YouTube Audio Books")
    parser.add_argument("url", help="YouTube video URL or channel URL")
    parser.add_argument("--max-videos", type=int, default=1, help="Max videos for channel")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--raw", default=None, help="Raw download directory")
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    args = parser.parse_args(argv)

    # Load config
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings.from_cli_overrides(
            raw_dir=args.raw,
            output_dir=args.output,
        )

    # Detect channel/playlist
    last_seg = args.url.rstrip("/").split("/")[-1]
    if "@" in last_seg or "/playlist" in args.url or "/@" in args.url:
        print(f"[Channel] Getting {args.max_videos} videos...")
        videos = get_channel_videos(args.url, args.max_videos)
        print(f"   Found {len(videos)} videos:")
        for v in videos:
            print(f"   - {v['title']}")
        all_results = []
        for v in videos:
            result = run_pipeline(
                v["url"], raw_dir=args.raw,
                output_dir=args.output, settings=settings,
            )
            all_results.append(result)
        print(f"\n{'='*60}")
        print("[SUMMARY]:")
        for r in all_results:
            print(f"   {r['video_id']}: {r['success']} segments")
    else:
        run_pipeline(
            args.url, raw_dir=args.raw,
            output_dir=args.output, settings=settings,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Simplify `pipeline_tts.py`**

Replace entire file below the docstring:

```python
"""
TTS Dataset Pipeline — YouTube audio books to TTS training datasets.

Usage:
  python pipeline_tts.py <video_url>
  python pipeline_tts.py <channel_url> --max-videos 3
  python pipeline_tts.py <video_url> --config configs/default.yaml
"""

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tts_pipeline.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test the refactored entry point**

```bash
cd D:/MinhTh_code/AudioProcess
rm -rf dataset/audio dataset/*.parquet raw/*
python pipeline_tts.py --help
```

Expected: help text is printed.

- [ ] **Step 4: Test full pipeline run**

```bash
cd D:/MinhTh_code/AudioProcess
python pipeline_tts.py "https://www.youtube.com/watch?v=iUGFXuxHNAA" --output dataset --raw raw 2>&1 | tail -10
```

Expected: pipeline runs, produces dataset with train/test split.

- [ ] **Step 5: Commit**

```bash
git add tts_pipeline/cli.py pipeline_tts.py
git commit -m "refactor: extract CLI module with config support"
```

---

### Task 7: Batch processing script

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\scripts\batch_process.py`
- Create: `D:\MinhTh_code\AudioProcess\scripts\__init__.py`

**Interfaces:**
- Consumes: `tts_pipeline.cli.run_pipeline` + list of URLs
- Produces: JSON summary file with per-video stats

- [ ] **Step 1: Write batch script**

Target file: `D:\MinhTh_code\AudioProcess\scripts\batch_process.py`

```python
"""Batch process multiple YouTube URLs and merge results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_pipeline.cli import run_pipeline
from tts_pipeline.config import Settings


def main():
    """Read URLs from stdin or file, process each, output summary."""
    if len(sys.argv) > 1 and sys.argv[1] == "--urls":
        urls = [sys.argv[2]]
    elif len(sys.argv) > 1 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        print("Usage: python scripts/batch_process.py --url <URL>")
        print("       python scripts/batch_process.py --file <urls.txt>")
        sys.exit(1)

    settings = Settings()
    results = []

    for url in urls:
        r = run_pipeline(url, settings=settings)
        results.append(r)

    summary_path = Path(settings.output_dir) / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = sum(r["success"] for r in results)
    failed = sum(r["fail"] for r in results)
    print(f"\nBatch complete: {total} samples, {failed} failures")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create __init__.py**

Target file: `D:\MinhTh_code\AudioProcess\scripts\__init__.py` (empty file).

- [ ] **Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: add batch processing script"
```

---

### Task 8: HuggingFace upload script

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\scripts\push_to_hub.py`

**Interfaces:**
- Consumes: Parquet path + WAV directory
- Produces: uploads to HuggingFace Hub

- [ ] **Step 1: Write push_to_hub script**

Target file: `D:\MinhTh_code\AudioProcess\scripts\push_to_hub.py`

```python
"""
Upload a TTS dataset to HuggingFace Hub.

Usage:
  python scripts/push_to_hub.py --parquet dataset/sample_dataset.parquet --repo user/my-tts-dataset
  python scripts/push_to_hub.py --dir sample_dataset/ --repo user/my-tts-dataset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from huggingface_hub import HfApi, upload_folder


def upload_parquet_to_hub(
    parquet_path: str,
    repo_id: str,
    token: str | None = None,
    commit_message: str = "Add TTS dataset",
) -> None:
    """Upload a single Parquet file to HF Hub as a dataset."""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=Path(parquet_path).name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def upload_dataset_dir_to_hub(
    dataset_dir: str,
    repo_id: str,
    token: str | None = None,
) -> None:
    """Upload an entire dataset directory (parquet + audio/) to HF Hub."""
    upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def create_dataset_card(repo_id: str, num_samples: int, language: str = "vi") -> str:
    """Generate a dataset card README.md for the HF Hub."""
    return f"""---
language: {language}
license: cc-by-nc-4.0
task:
  - text-to-speech
---

# TTS Dataset — {repo_id}

## Description
Vietnamese TTS dataset generated from YouTube audio books.

## Statistics
- Samples: {num_samples}
- Format: Parquet + WAV (mono, 22050Hz, 16-bit)

## Columns
| Column | Type | Description |
|--------|------|-------------|
| `audio` | string | Relative path to WAV file |
| `audioduration (s)` | float | Duration in seconds |
| `transcription` | string | Vietnamese text transcription |
| `file_name` | string | WAV filename |

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("{repo_id}")
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload TTS dataset to HuggingFace Hub")
    parser.add_argument("--parquet", help="Path to Parquet file")
    parser.add_argument("--dir", help="Path to dataset directory (parquet + audio/)")
    parser.add_argument("--repo", required=True, help="HF repo ID (user/repo-name)")
    parser.add_argument("--token", help="HF token (default: use huggingface-cli login)")

    args = parser.parse_args()

    if args.parquet:
        upload_parquet_to_hub(args.parquet, args.repo, args.token)
        df = pd.read_parquet(args.parquet)
        card = create_dataset_card(args.repo, len(df))
        HfApi(token=args.token).upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print(f"[OK] Uploaded {args.parquet} to {args.repo}")
    elif args.dir:
        upload_dataset_dir_to_hub(args.dir, args.repo, args.token)
        print(f"[OK] Uploaded {args.dir} to {args.repo}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/push_to_hub.py
git commit -m "feat: add HuggingFace upload script"
```

---

### Task 9: Update SPEC.md and add CLAUDE.md

**Files:**
- Create: `D:\MinhTh_code\AudioProcess\CLAUDE.md`
- Modify: `D:\MinhTh_code\AudioProcess\SPEC.md` (add train/test split and config sections)

- [ ] **Step 1: Create CLAUDE.md**

Target file: `D:\MinhTh_code\AudioProcess\CLAUDE.md`

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

# Run pipeline on a single video
python pipeline_tts.py <video_url> [--output dir] [--raw dir] [--config config.yaml]

# Run pipeline on a channel (first N videos)
python pipeline_tts.py <channel_url> --max-videos 5

# Batch process
python scripts/batch_process.py --file urls.txt

# Upload to HuggingFace
python scripts/push_to_hub.py --dir sample_dataset/ --repo user/dataset-name

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
| `audio` | string | Relative path to WAV |
| `audioduration (s)` | float | Duration (2-20s) |
| `transcription` | string | Vietnamese text |
| `file_name` | string | WAV filename |

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
```

- [ ] **Step 2: SPEC.md — add train/test split note**

Insert into section 3.4 after the existing text:

```markdown
### 3.5. Train/test split

Khi export, dataset được chia train/test theo tỉ lệ `train_split` (default 0.9).
Output gồm 2 file Parquet: `{video_id}_train.parquet` và `{video_id}_test.parquet`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md SPEC.md
git commit -m "docs: add CLAUDE.md, update SPEC.md with train/test split"
```

---

### Self-Review Checklist

1. **Spec coverage:** Does the plan cover every item in SPEC.md §5 "Có thể cải thiện"?
   - VAD: ❌ (deferred — too complex for this phase, requires model dependency)
   - Whisper fallback: ❌ (deferred — current pipeline depends on YouTube captions)
   - Audio augmentation: ❌ (deferred — out of scope for dataset creation)
   - **Train/test split: ✅ (Task 5)**
   - CER/WER validation: ❌ (deferred)
   - UI: ❌ (deferred)
   - **Config/HF upload/Batch: ✅ (Tasks 1, 7, 8)**
   - **Refactor: ✅ (Tasks 2-6)**

2. **Placeholder scan:** No "TBD", "TODO", or placeholder code in any task.

3. **Type consistency:** All functions use the same return types across tasks.
   - `download_video_and_subs` returns `{video_id, title, audio_path, vtt_path}` — consistent.
   - Parsed segments are `{start, end, text}` — consistent.
   - `export_dataset` returns `{video_id, total_segments, success, fail, parquet_paths}` — consistent.

---

### Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-23-tts-pipeline-v2.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
