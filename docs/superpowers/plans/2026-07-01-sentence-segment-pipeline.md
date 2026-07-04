# Sentence-Level Segmentation & Pipeline V2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Mỗi segment = đúng 1 câu (hoặc gộp vài câu ngắn), không overlap thời gian, không segment quá ngắn.

**Architecture:** Pipeline mới: `merge → split_by_sentence → dedup → fix_time_overlaps → merge_short_segments → filter → export`

**Tech Stack:** Python 3.12+, pyarrow, ffmpeg

---

### Task 1: Cập nhật `processor.py` — thêm 3 hàm mới

**Files:**
- Modify: `tts_pipeline/processor.py`
- Test: `tests/test_processor.py`

**Nội dung:**
- Giữ nguyên `clean_text()`, `dedup_consecutive_text()`, `split_by_sentence()`
- Thêm `fix_time_overlaps()` — xoá overlap thời gian
- Thêm `merge_short_segments()` — gộp câu ngắn

- [ ] **Step 1: Viết test cho các hàm mới**

Thêm vào `tests/test_processor.py`:

```python
def test_fix_time_overlaps_pushes_start():
    segs = [
        {"start": 0, "end": 5, "text": "Đoạn một."},
        {"start": 3, "end": 8, "text": "Đoạn hai."},
    ]
    result = fix_time_overlaps(segs)
    assert len(result) == 2
    assert result[1]["start"] >= result[0]["end"]


def test_fix_time_overlaps_removes_too_short():
    segs = [
        {"start": 0, "end": 5, "text": "Đoạn một."},
        {"start": 5, "end": 5.05, "text": "quá ngắn"},
    ]
    result = fix_time_overlaps(segs)
    assert len(result) == 1


def test_merge_short_merges_into_next():
    segs = [
        {"start": 0, "end": 3, "text": "Đoạn ngắn"},
        {"start": 3, "end": 10, "text": "Đoạn dài. tiếp theo."},
    ]
    result = merge_short_segments(segs, min_dur=4.0, min_text_len=5)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 10


def test_merge_short_keeps_normal():
    segs = [
        {"start": 0, "end": 5, "text": "Đoạn bình thường."},
        {"start": 5, "end": 10, "text": "Đoạn tiếp theo."},
    ]
    result = merge_short_segments(segs, min_dur=2.0, min_text_len=5)
    assert len(result) == 2
```

- [ ] **Step 2: Chạy test — phải fail vì chưa có hàm**

```bash
cd D:/MinhTh_code/DataAudioProcess
python -m pytest tests/test_processor.py -v
```
Expected: 4 tests ImportError (chưa có `fix_time_overlaps`, `merge_short_segments`).

- [ ] **Step 3: Thêm 2 hàm vào `tts_pipeline/processor.py`**

```python
def fix_time_overlaps(segments: list[dict]) -> list[dict]:
    """Fix overlapping time between consecutive segments."""
    result: list[dict] = []
    for seg in segments:
        if result:
            prev_end = result[-1]["end"]
            if seg["start"] < prev_end:
                seg = {**seg, "start": prev_end}
            if seg["end"] - seg["start"] < 0.1:
                continue
        result.append(seg)
    return result


def merge_short_segments(
    segments: list[dict],
    min_dur: float = 2.0,
    min_text_len: int = 10,
) -> list[dict]:
    """Merge short sentences into the following segment.

    A segment is 'short' if duration < min_dur or text length < min_text_len.
    Accumulates consecutive short segments, then merges the whole block
    into the first non-short segment that follows.
    """
    if not segments:
        return []

    result: list[dict] = []
    buffer = None

    for seg in segments:
        dur = seg["end"] - seg["start"]
        text_len = len(seg["text"])
        is_short = dur < min_dur or text_len < min_text_len

        if is_short and buffer is None:
            buffer = seg
        elif is_short and buffer is not None:
            gap = seg["start"] - buffer["end"]
            spacer = " " if gap < 2.0 else ". "
            buffer = {
                "start": buffer["start"],
                "end": seg["end"],
                "text": buffer["text"].rstrip() + spacer + seg["text"],
            }
        elif buffer is not None:
            gap = seg["start"] - buffer["end"]
            spacer = " " if gap < 2.0 else ". "
            merged = {
                "start": buffer["start"],
                "end": seg["end"],
                "text": buffer["text"].rstrip() + spacer + seg["text"],
            }
            result.append(merged)
            buffer = None
        else:
            result.append(seg)

    if buffer is not None:
        result.append(buffer)

    return result
```

- [ ] **Step 4: Chạy test — confirm pass**

```bash
cd D:/MinhTh_code/DataAudioProcess
python -m pytest tests/ -v
```
Expected: 17 tests pass.

---

### Task 2: Cập nhật `cli.py` — pipeline order mới

**Files:**
- Modify: `tts_pipeline/cli.py`

- [ ] **Step 1: Cập nhật import**

```python
from tts_pipeline.processor import clean_text, split_by_sentence, dedup_consecutive_text, fix_time_overlaps, merge_short_segments
```

- [ ] **Step 2: Cập nhật pipeline trong `run_pipeline()`**

```python
    for seg in merged:
        seg["text"] = clean_text(seg["text"])

    merged = split_by_sentence(merged, s.min_segment_dur)
    print(f"   After sentence split: {len(merged)}")

    merged = dedup_consecutive_text(merged)
    merged = fix_time_overlaps(merged)
    merged = merge_short_segments(merged, s.min_segment_dur, s.min_text_len)
    print(f"   After dedup + time fix + merge short: {len(merged)}")

    merged = [seg for seg in merged if seg["text"] and len(seg["text"]) >= s.min_text_len
              and (seg["end"] - seg["start"]) >= s.min_segment_dur]
    print(f"   After filter: {len(merged)}")
```

- [ ] **Step 3: Chạy test**

```bash
cd D:/MinhTh_code/DataAudioProcess
python -m pytest tests/ -v
```
Expected: 17 tests pass.

---

### Task 3: Chạy thử + verify

- [ ] **Step 1: Test pipeline với data thật**

```bash
cd D:/MinhTh_code/DataAudioProcess
python -c "
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.processor import clean_text, split_by_sentence, dedup_consecutive_text, fix_time_overlaps, merge_short_segments
from tts_pipeline.config import Settings

s = Settings()
merged = merge_cues(parse_vtt_full('raw/iUGFXuxHNAA.vi.vtt'), s)
for seg in merged:
    seg['text'] = clean_text(seg['text'])

merged = split_by_sentence(merged, s.min_segment_dur)
merged = dedup_consecutive_text(merged)
merged = fix_time_overlaps(merged)
merged = merge_short_segments(merged, s.min_segment_dur, s.min_text_len)
merged = [seg for seg in merged if seg['text'] and len(seg['text']) >= s.min_text_len
          and (seg['end'] - seg['start']) >= s.min_segment_dur]

overlaps = sum(1 for i in range(len(merged)-1) if merged[i]['end'] > merged[i+1]['start'])
short = sum(1 for s in merged if s['end']-s['start'] < s.min_segment_dur or len(s['text']) < s.min_text_len)
total_dur = sum(s['end'] - s['start'] for s in merged)
print(f'Segments: {len(merged)}')
print(f'Overlaps: {overlaps}')
print(f'Short: {short}')
print(f'Total dur: {total_dur/60:.1f} min')
print()
for i, seg in enumerate(merged[:15]):
    dur = seg['end'] - seg['start']
    print(f'{i:3d} [{seg[\"start\"]:6.1f}-{seg[\"end\"]:6.1f}] ({dur:4.1f}s len={len(seg[\"text\"]):3d}) {seg[\"text\"][:80]}')
"
```

Expected: Overlaps = 0, Short = 0, mỗi segment là 1 câu hoặc gộp ngắn hợp lý.

- [ ] **Step 2: Chạy full pipeline + export**

```bash
cd D:/MinhTh_code/DataAudioProcess
python pipeline_tts.py "https://www.youtube.com/watch?v=iUGFXuxHNAA" --output dataset --raw raw
```

- [ ] **Step 3: Verify Parquet schema**

```bash
cd D:/MinhTh_code/DataAudioProcess
python -c "
import pyarrow.parquet as pq
s = pq.read_schema('dataset/iUGFXuxHNAA_train.parquet')
print(s)
table = pq.read_table('dataset/iUGFXuxHNAA_train.parquet')
print(f'Rows: {len(table)}')
for i in range(5):
    t = table.column('transcription')[i].as_py()
    print(f'{i}: {t[:80]}')
"
```

- [ ] **Step 4: Upload lên HF**

```bash
python scripts/push_to_hub.py --config configs/upload.yaml
```

---

### Task 4: Cập nhật docs

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `CLAUDE.md`
- Modify: `docs/sentence-segment-spec.md`

- [ ] **Step 1: Cập nhật README.md**

Thêm mô tả về sentence-level segmentation, pipeline order mới.

- [ ] **Step 2: Cập nhật SPEC.md**

Sửa pipeline flow diagram.

- [ ] **Step 3: Cập nhật CLAUDE.md**

Thêm commands/tests cho các hàm mới.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: sentence-level segmentation with short segment merging

- Split every segment into individual sentences
- Pipeline: merge -> split_by_sentence -> dedup -> fix_time_overlaps -> merge_short_segments -> filter
- fix_time_overlaps() removes time range overlaps
- merge_short_segments() merges short sentences into the next segment
- Remove split_long_segments (replaced by split_by_sentence)

Co-Authored-By: Claude <noreply@anthropic.com>
"
```
