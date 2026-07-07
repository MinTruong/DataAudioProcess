# VAD Re-segment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VTT timing with Silero-VAD speech detection for accurate segment boundaries.

**Architecture:** Silero-VAD quét audio WAV thành speech intervals → gộp 5-20s → align text từ raw VTT cues bằng temporal overlap.

**Tech Stack:** Python, silero-vad, onnxruntime, librosa

## Global Constraints

- Must work on CPU. No GPU required.
- VAD model auto-downloaded on first use (cached at `~/.cache/silero-vad/`).
- Must handle 45-min audio within 3 min processing time.
- Text source is still YouTube VTT — alignment by temporal overlap, not ASR.
- Fallback to original VTT timing if VAD returns no intervals.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tts_pipeline/config.py` | Modify | Add `vad_enabled`, `vad_threshold`, `vad_min_speech_dur`, `vad_min_silence_dur` |
| `tts_pipeline/vad.py` | **Create** | Silero-VAD wrapper: load model, scan audio → `[{start, end}]` |
| `tts_pipeline/vad_aligner.py` | **Create** | Group intervals 5-20s + align text from raw VTT cues |
| `tts_pipeline/cli.py` | Modify | Integrate VAD flow, remove segment_by_content |
| `tests/test_vad.py` | **Create** | Unit tests for vad.py |
| `tests/test_vad_aligner.py` | **Create** | Unit tests for vad_aligner.py |
| `requirements.txt` | Modify | Add `silero-vad>=0.2`, `onnxruntime>=1.15`, `librosa>=0.10` |

---

### Task 1: Settings + dependencies

**Files:**
- Modify: `tts_pipeline/config.py` (thêm 4 VAD fields)
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: existing `Settings` class
- Produces: `Settings` with VAD fields

- [ ] **Step 1: Thêm VAD fields vào config.py**

```python
# After train_split
vad_enabled: bool = Field(default=True, description="Enable Silero-VAD for speech boundary detection")
vad_threshold: float = Field(default=0.5, ge=0.1, le=0.9, description="VAD speech probability threshold")
vad_min_speech_dur: float = Field(default=0.3, ge=0.1, le=2.0, description="Min speech duration (s) to keep")
vad_min_silence_dur: float = Field(default=0.3, ge=0.1, le=5.0, description="Min silence (s) to split intervals")
```

- [ ] **Step 2: Cập nhật requirements.txt**

```
# Thêm cuối file
silero-vad>=0.2
onnxruntime>=1.15
librosa>=0.10
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: add VAD settings to config"
```

---

### Task 2: vad.py — Silero-VAD scanner

**Files:**
- Create: `tts_pipeline/vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Produces: `get_speech_intervals(audio_path, sample_rate, threshold, min_speech_dur, min_silence_dur) -> list[dict]`

- [ ] **Step 1: Write failing test**

```python
"""Tests for VAD speech interval detection."""
import numpy as np
import soundfile as sf
from pathlib import Path
from tts_pipeline.vad import get_speech_intervals


def _create_test_audio(duration: float, sample_rate: int = 22050, add_silence: bool = False) -> str:
    """Create a test WAV with tone + optional silence gaps."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz tone
    if add_silence:
        # Insert 1s silence in the middle
        silence_len = int(sample_rate * 1.0)
        audio[silence_len:-silence_len] = 0
    path = "/tmp/_test_vad.wav"
    sf.write(path, audio, sample_rate)
    return path


def test_vad_returns_intervals():
    """get_speech_intervals returns a list of {start, end} dicts."""
    path = _create_test_audio(3.0)
    intervals = get_speech_intervals(path)
    assert isinstance(intervals, list)
    assert len(intervals) > 0
    for iv in intervals:
        assert "start" in iv and "end" in iv
        assert iv["end"] > iv["start"]


def test_vad_detects_silence_gap():
    """Intervals should be split at the silence gap."""
    path = _create_test_audio(5.0, add_silence=True)
    intervals = get_speech_intervals(path, min_silence_dur=0.5)
    # With 1s silence in middle, should get 2 intervals
    assert len(intervals) >= 2, f"Expected >=2 intervals, got {len(intervals)}"


def test_vad_returns_empty_for_silence():
    """All-silence audio returns empty list."""
    sample_rate = 22050
    audio = np.zeros(int(sample_rate * 2.0))
    path = "/tmp/_test_vad_silence.wav"
    sf.write(path, audio, sample_rate)
    intervals = get_speech_intervals(path)
    assert intervals == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vad.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write implementation**

```python
"""Silero-VAD speech interval detection."""
from pathlib import Path

import librosa
import numpy as np
import torch

# Silero-VAD auto-downloads model on first import
MODEL, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
)


def get_speech_intervals(
    audio_path: str | Path,
    sample_rate: int = 22050,
    threshold: float = 0.5,
    min_speech_dur: float = 0.3,
    min_silence_dur: float = 0.3,
) -> list[dict]:
    """Scan audio with Silero-VAD, return list of {start, end} speech intervals.

    Steps:
    1. Load audio via librosa (mono, resample to 16000 for VAD)
    2. Process in 30ms frames → speech probabilities
    3. Threshold at ``threshold`` → binary voice flag per frame
    4. Gộp consecutive speech frames >= ``min_speech_dur``
    5. Gộp intervals cách nhau < ``min_silence_dur``
    6. Scale timestamps back to input ``sample_rate`` timebase
    """
    # Load audio
    audio, orig_sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Resample to 16kHz for VAD (Silero expects 16k)
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        vad_sr = 16000
    else:
        audio_16k = audio
        vad_sr = 16000

    # Process with Silero-VAD
    frames = []
    frame_len = int(vad_sr * 0.03)  # 30ms per frame
    for i in range(0, len(audio_16k), frame_len):
        chunk = audio_16k[i: i + frame_len]
        if len(chunk) < frame_len:
            chunk = np.pad(chunk, (0, frame_len - len(chunk)), "constant")
        speech_prob = MODEL(torch.from_numpy(chunk), vad_sr).item()
        frames.append(speech_prob > threshold)

    # Build raw intervals from consecutive speech frames
    raw_intervals = []
    in_speech = False
    start_frame = 0
    for i, is_speech in enumerate(frames):
        if is_speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not is_speech and in_speech:
            dur = (i - start_frame) * 0.03
            if dur >= min_speech_dur:
                raw_intervals.append({
                    "start": start_frame * 0.03 * (sample_rate / vad_sr),
                    "end": i * 0.03 * (sample_rate / vad_sr),
                })
            in_speech = False
    if in_speech:
        dur = (len(frames) - start_frame) * 0.03
        if dur >= min_speech_dur:
            raw_intervals.append({
                "start": start_frame * 0.03 * (sample_rate / vad_sr),
                "end": len(frames) * 0.03 * (sample_rate / vad_sr),
            })

    # Merge intervals separated by < min_silence_dur
    if not raw_intervals:
        return []
    merged = [raw_intervals[0]]
    for iv in raw_intervals[1:]:
        gap = iv["start"] - merged[-1]["end"]
        if gap < min_silence_dur:
            merged[-1]["end"] = iv["end"]
        else:
            merged.append(iv)

    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vad.py -v`
Expected: PASS (may need 1-2s for model download first time)

- [ ] **Step 5: Commit**

```bash
git add tests/test_vad.py tts_pipeline/vad.py
git commit -m "feat: add Silero-VAD speech interval scanner"
```

---

### Task 3: vad_aligner.py — group intervals + text alignment

**Files:**
- Create: `tts_pipeline/vad_aligner.py`
- Test: `tests/test_vad_aligner.py`

**Interfaces:**
- Consumes: `get_speech_intervals()` → `[{start, end}]`, `parse_vtt_full()` → `[{start, end, text}]`
- Produces: `group_vad_intervals(intervals, min_dur=5.0, max_dur=20.0) -> list[dict]`
- Produces: `align_text_to_groups(groups, raw_vtt_cues, punctuation=True) -> list[dict]`

- [ ] **Step 1: Write failing test**

```python
"""Tests for VAD interval grouping and text alignment."""
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups


def test_group_empty():
    assert group_vad_intervals([]) == []


def test_group_single():
    iv = [{"start": 0, "end": 3}]
    result = group_vad_intervals(iv, min_dur=5.0, max_dur=20.0)
    # Single short interval stays as-is (tail emit)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 3


def test_group_multiple():
    iv = [
        {"start": 0, "end": 8},
        {"start": 10, "end": 14},
        {"start": 16, "end": 22},
    ]
    result = group_vad_intervals(iv, min_dur=5.0, max_dur=12.0)
    assert len(result) >= 2  # first two (8s+4s=12s max), third is tail


def test_align_text_basic():
    groups = [{"start": 5, "end": 15}]
    cues = [
        {"start": 4, "end": 10, "text": "xin chào"},
        {"start": 10, "end": 16, "text": "các bạn"},
    ]
    result = align_text_to_groups(groups, cues, punctuation=False)
    assert len(result) == 1
    assert "xin chào" in result[0]["text"]
    assert "các bạn" in result[0]["text"]
    assert result[0]["start"] == 5
    assert result[0]["end"] == 15


def test_align_no_overlap():
    groups = [{"start": 100, "end": 110}]
    cues = [{"start": 0, "end": 5, "text": "hello"}]
    result = align_text_to_groups(groups, cues, punctuation=False)
    assert len(result) == 1
    assert result[0]["text"] == ""


def test_align_partial_overlap():
    groups = [{"start": 2, "end": 8}]
    cues = [
        {"start": 0, "end": 4, "text": "alo"},    # 50% overlap
        {"start": 4, "end": 5, "text": "xin"},    # full overlap
        {"start": 9, "end": 12, "text": "chào"},  # no overlap (gap)
    ]
    result = align_text_to_groups(groups, cues, punctuation=False)
    assert "alo" in result[0]["text"]
    assert "xin" in result[0]["text"]
    assert "chào" not in result[0]["text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vad_aligner.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write implementation**

```python
"""VAD interval grouping and text alignment."""
import re

from tts_pipeline.punctuator import restore_punctuation


def group_vad_intervals(
    intervals: list[dict],
    min_dur: float = 5.0,
    max_dur: float = 20.0,
) -> list[dict]:
    """Gộp speech intervals thành group 5-20s.

    Greedy: gom interval cho đến ``max_dur``, emit group, reset.
    Tail luôn được emit.
    """
    if not intervals:
        return []

    groups: list[list[dict]] = []
    current = [intervals[0]]

    for iv in intervals[1:]:
        group_dur = current[-1]["end"] - current[0]["start"]
        iv_dur = iv["end"] - iv["start"]

        if group_dur < min_dur:
            current.append(iv)
        elif group_dur + iv_dur > max_dur and group_dur >= min_dur:
            groups.append(current)
            current = [iv]
        else:
            current.append(iv)

    groups.append(current)  # always emit tail

    return [
        {"start": g[0]["start"], "end": g[-1]["end"]}
        for g in groups
    ]


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Fraction of (a) that overlaps with (b)."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    duration = a_end - a_start
    if duration <= 0:
        return 0.0
    return overlap / duration


def align_text_to_groups(
    groups: list[dict],
    raw_vtt_cues: list[dict],
    punctuation: bool = True,
) -> list[dict]:
    """Gán text cho mỗi VAD group từ raw VTT cues overlap.

    Với mỗi group:
    1. Tìm raw VTT cues có overlap > 0.3 với group
    2. Lấy text, join, clean whitespace
    3. Nếu punctuation=True, chạy restore_punctuation()
    4. Return {start, end, text}
    """
    results = []
    for g in groups:
        texts = []
        for cue in raw_vtt_cues:
            ratio = _overlap_ratio(
                cue["start"], cue["end"],
                g["start"], g["end"],
            )
            if ratio > 0.3:
                texts.append(cue["text"])

        joined = " ".join(texts)
        joined = re.sub(r"\s+", " ", joined).strip()

        if punctuation and joined:
            joined = restore_punctuation(joined)

        results.append({
            "start": g["start"],
            "end": g["end"],
            "text": joined,
        })

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vad_aligner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_vad_aligner.py tts_pipeline/vad_aligner.py
git commit -m "feat: add VAD interval grouper and text aligner"
```

---

### Task 4: cli.py — integrate VAD flow

**Files:**
- Modify: `tts_pipeline/cli.py`

**Interfaces:**
- Consumes: `get_speech_intervals()`, `group_vad_intervals()`, `align_text_to_groups()`

- [ ] **Step 1: Sửa cli.py — tích hợp VAD**

Thay phần `segment_by_content` trong `run_pipeline()`:

```python
# Step 2: Parse + merge cues
print("\n[Step 2] Parse caption & merge cues...")
raw_cues = parse_vtt_full(video_info["vtt_path"])
print(f"   Raw cues: {len(raw_cues)}")
merged = merge_cues(raw_cues, s)
print(f"   After merge: {len(merged)}")

for seg in merged:
    seg["text"] = clean_text(seg["text"])

# Punctuation restoration
from tts_pipeline.punctuator import _split_segment as _do_split
merged = [_do_split(s)[0] for s in merged]

# Step 3: VAD-based re-segment (thay segment_by_content)
print("\n[Step 3] VAD speech detection...")
from tts_pipeline.vad import get_speech_intervals
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups

if s.vad_enabled:
    speech = get_speech_intervals(
        video_info["audio_path"],
        sample_rate=s.sample_rate,
        threshold=s.vad_threshold,
        min_speech_dur=s.vad_min_speech_dur,
        min_silence_dur=s.vad_min_silence_dur,
    )
    print(f"   Speech intervals: {len(speech)}")

    groups = group_vad_intervals(speech, s.min_segment_dur, s.max_segment_dur)
    print(f"   After grouping: {len(groups)}")

    segments = align_text_to_groups(groups, raw_cues, punctuation=True)
    print(f"   After text alignment: {len(segments)}")
else:
    # Fallback: original segment_by_content
    merged = fix_time_overlaps(merged)
    segments = segment_by_content(merged, s.min_segment_dur, s.max_segment_dur)
    segments = fix_time_overlaps(segments)

# Filter
segments = [seg for seg in segments
            if seg["text"]
            and len(seg["text"]) >= s.min_text_len
            and (seg["end"] - seg["start"]) >= s.min_segment_dur]
print(f"   After filter: {len(segments)}")

# Step 4: Export
print("\n[Step 4] Cut audio & export dataset...")
result = export_dataset(
    segments, out, video_info["video_id"],
    video_info["audio_path"], s,
    split_train_test=True,
)
```

Xóa imports không còn dùng:
- `fix_time_overlaps` (chỉ dùng trong fallback path)
- `segment_by_content` (chỉ dùng trong fallback path)

Giữ `dedup_consecutive_text` nhưng không gọi (sẵn cho fallback).

- [ ] **Step 2: Run pipeline test**

Run: `python pipeline_tts.py https://www.youtube.com/watch?v=T_zgDuLSIYU --raw raw --output dataset_test 2>&1 | head -20`
Expected: Steps 1-4 complete, VAD scan chạy

- [ ] **Step 3: Commit**

```bash
git add tts_pipeline/cli.py
git commit -m "feat: integrate VAD re-segment into pipeline"
```

---

### Task 5: Full integration test + cleanup

**Files:**
- Test: run full pipeline on test video

- [ ] **Step 1: Run full pipeline**

```bash
rm -rf dataset_test/raw/T_zgDuLSIYU*
python pipeline_tts.py "https://www.youtube.com/watch?v=T_zgDuLSIYU" --raw raw --output dataset_test
```

Check output: `python -c "import pyarrow.parquet as pq; t=pq.read_table('dataset_test/T_zgDuLSIYU_train.parquet'); print(len(t))"`

- [ ] **Step 2: Check segment quality**

```bash
python -c "
import pyarrow.parquet as pq, io, wave, statistics
tab = pq.read_table('dataset_test/T_zgDuLSIYU_train.parquet')
audios = tab.column('audio').to_pylist()
durs = []
for a in audios:
    with io.BytesIO(a['bytes']) as buf:
        with wave.open(buf) as w:
            durs.append(w.getnframes()/w.getframerate())
print(f'{len(tab)} rows, mean={statistics.mean(durs):.1f}s, <5={len([d for d in durs if d<5.0])}, >20={len([d for d in durs if d>20.0])}')
"
```

Expected: Mean ~10-15s, 0 <5s, few or 0 >20s

- [ ] **Step 3: Run pytest**

```bash
python -m pytest tests/ -v
```

Expected: 33+ tests pass (new VAD tests + existing)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: full VAD re-segment integration"
```
