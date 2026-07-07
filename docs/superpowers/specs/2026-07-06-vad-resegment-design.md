# VAD Re-segment Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:writing-plans → superpowers:subagent-driven-development to implement this spec task-by-task.

**Goal:** Replace VTT timing with Silero-VAD speech detection for accurate segment boundaries in TTS dataset pipeline.

**Architecture:** Silero-VAD quét audio WAV thành speech intervals → gộp 5-20s → align text từ VTT cues → xuất dataset.

**Tech Stack:** Python, silero-vad, onnxruntime, numpy (existing)

## Global Constraints

- Must work on CPU (device configurable). No GPU required.
- VAD model downloaded once, cached at `~/.cache/silero-vad/`.
- Must handle 45-min audio files within 3 min processing time.
- Must not add >200ms latency per segment at export time.
- Text source is still YouTube VTT — alignment is by temporal overlap, not ASR.

---

## Architecture

```
YouTube URL
    │
    ▼
┌────────────────────┐
│  downloader        │  yt-dlp: audio.wav + vi.vtt
└──────┬─────────────┘
       │
       ▼
┌─────────────────────────────┐
│  parser (unchanged)         │
│  parse_vtt_full()           │  raw cues [{start,end,text}]
│  merge_cues()               │  merged segments (for text)
└──────┬──────────────────────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐
│ merged segments  │    │ Silero-VAD Scanner   │ NEW
│ (text only)      │    │ → speech_intervals[] │
└─────────────────┘    └──────────────────────┘
       │                          │
       └──────────┬───────────────┘
                  ▼
┌───────────────────────────────┐
│ VAD Aligner                   │ NEW
│ group VAD intervals 5-20s     │
│ align text from raw VTT cues  │
│ restore punctuation           │
└───────────────────────────────┘
                  │
                  ▼
┌────────────────────┐
│ filter + exporter  │  → Parquet + WAV
└────────────────────┘
```

## Module: vad.py

**File:** `tts_pipeline/vad.py`

**Responsibility:** Load Silero-VAD model, scan audio WAV, return speech intervals.

```python
def get_speech_intervals(
    audio_path: str,
    sample_rate: int = 22050,
    threshold: float = 0.5,
    min_speech_dur: float = 0.3,
    min_silence_dur: float = 0.3,
) -> list[dict]:
    """Scan audio with Silero-VAD, return list of {start, end} speech intervals.

    Steps:
    1. Load audio via librosa/soundfile (mono, 22050Hz)
    2. Load Silero-VAD ONNX model
    3. Process in 30ms frames → speech probabilities
    4. Threshold at 0.5 → binary voice flag per frame
    5. Gộp consecutive speech frames >= min_speech_dur
    6. Gộp intervals cách nhau < min_silence_dur
    7. Return [{start, end}, ...]
    """
```

Config mới trong `Settings`:
```python
vad_enabled: bool = True
vad_threshold: float = 0.5
vad_min_speech_dur: float = 0.3
vad_min_silence_dur: float = 0.3
```

**Dependencies:** `silero-vad>=0.2`, `onnxruntime>=1.15`, `librosa>=0.10`

## Module: vad_aligner.py

**File:** `tts_pipeline/vad_aligner.py`

**Responsibility:** Gộp VAD intervals → segments 5-20s, align text từ raw VTT cues.

```python
def group_vad_intervals(
    intervals: list[dict],
    min_dur: float = 5.0,
    max_dur: float = 20.0,
) -> list[dict]:
    """Gộp speech intervals thành group 5-20s.

    Greedy: gom interval cho đến max_dur, emit group, reset.
    Tail luôn được emit.
    """
```

```python
def align_text_to_groups(
    groups: list[dict],
    raw_vtt_cues: list[dict],
    punctuation: bool = True,
) -> list[dict]:
    """Gán text cho mỗi VAD group từ raw VTT cues overlap.

    Với mỗi group:
    1. Tìm raw VTT cues có overlap với group[start, end]
    2. Filter cues có overlap_ratio > 0.3
    3. Lấy text, join, clean, punctuate
    4. Return {start, end, text}

    Args:
        groups: [{start, end}, ...] từ group_vad_intervals()
        raw_vtt_cues: [{start, end, text}, ...] từ parse_vtt_full()
        punctuation: Nếu True, chạy restore_punctuation()

    Returns:
        list of {start, end, text}
    """
```

## Pipeline flow (cli.py)

```python
# Step 1: Download → parse_vtt_full → merge_cues (text only)
raw_cues = parse_vtt_full(vtt_path)          # [{start,end,text}]
merged = merge_cues(raw_cues, s)             # text segments

# Step 2: VAD scan (thay cho segment_by_content timing)
from tts_pipeline.vad import get_speech_intervals
speech = get_speech_intervals(audio_path)

# Step 3: Group VAD intervals → 5-20s
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups
groups = group_vad_intervals(speech, s.min_segment_dur, s.max_segment_dur)

# Step 4: Align text using raw VTT cues (not merged — raw có timing gốc)
segments = align_text_to_groups(groups, raw_cues)

# Step 5: filter + export
segments = [seg for seg in segments if seg["text"] and ...]
export_dataset(segments, ...)
```

## Edge cases

- **Silence music:** Nếu audio có nhạc nền, VAD có thể không detect được silence. Giải pháp: threshold tune, hoặc pre-filter energy < -40dB.
- **Very short speech burst:** "Ừ", "À" — intervals < 0.3s bị filter, text vẫn được gán vào group kế.
- **VTT cue lệch timing:** Alignment by overlap ratio > 0.3 — nếu cue overlap ít hơn, bỏ qua, tránh text sai.
- **Empty VAD:** Nếu không detect speech, fallback về VTT timing.

## Testing

- `tests/test_vad.py`: test VAD scan on known audio with/without silence
- `tests/test_vad_aligner.py`: test group + text alignment
- Integration: run pipeline, compare segment count & duration distribution
