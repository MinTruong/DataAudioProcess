# SPEC: Sentence-Level Segmentation cho TTS Dataset

## 1. Vấn đề

Pipeline hiện tại cắt segment dựa trên VTT cues của YouTube, sau khi merge vẫn còn **nhiều câu trong 1 segment** (chỉ split khi >20s). Điều này gây ra:
- Một segment chứa 2-3 câu không liên quan
- Thời gian cắt không chính xác
- Khó training TTS vì alignment audio-text không chuẩn

## 2. Mục tiêu

- Mỗi segment output = **1 câu** hoặc nhiều câu ngắn được gộp lại để đạt độ dài tối thiểu
- Không có segment <2s hoặc text <10 ký tự
- Không có overlap thời gian giữa các segment

## 3. Pipeline mới

```
YouTube URL
    │
    ▼
┌──────────────┐
│  downloader  │  yt-dlp: audio + VTT caption
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   parser     │  parse_vtt_full() → merge_cues()
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  clean_text  │  normalize whitespace
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ split_by_sentence │  tách từng câu, prorate time
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ dedup_consecutive │  xoá overlap text giữa segment kế
│ _text             │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ fix_time_overlaps │  đẩy start time nếu overlap
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ merge_short_seg-  │  gộp câu ngắn vào câu kế tiếp
│ ments             │
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│   filter     │  min/max text length + duration
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  exporter    │  ffmpeg cut + Parquet
└──────────────┘
```

## 4. Chi tiết xử lý

### 4.1. split_by_sentence()

- Dùng regex `(?<=[.!?])\s+` để tách theo dấu câu (`. ! ?`), fallback `,`
- **Prorate time**: chia tổng thời gian cho từng câu theo tỉ lệ ký tự
  - `char_ratio = total_duration / total_chars`
  - Mỗi câu: `duration = max(len(sentence) * char_ratio, min_dur)`
- Áp dụng cho **mọi segment** (không chỉ >20s)

### 4.2. dedup_consecutive_text()

Chạy **sau** split_by_sentence. Xoá overlap text 1-3 từ giữa segment kế:
- Check 3-word → 2-word → 1-word
- Nếu text rỗng → bỏ segment
- Viết hoa chữ cái đầu sau khi xoá

### 4.3. fix_time_overlaps()

- Duyệt từng segment, nếu `start(N+1) < end(N)` → set `start(N+1) = end(N)`
- Nếu `end - start < 0.1s` → bỏ segment

### 4.4. merge_short_segments() — MỚI

Gộp câu ngắn vào câu kế tiếp để tránh segment quá ngắn:

```python
def merge_short_segments(segments, min_dur=2.0, min_text_len=10):
```

- Segment được coi là **ngắn** nếu `duration < min_dur` hoặc `text_length < min_text_len`
- Accumulate các segment ngắn liên tiếp vào buffer
- Khi gặp segment đủ dài → gộp buffer + segment đó thành 1 segment
- Nhiều segment ngắn liên tiếp → gộp thành 1 segment dài
- Segment cuối cùng luôn được giữ nguyên

**Cách nối text:**
- Gap < 2.0s giữa các segment → nối bằng `" "` (cùng luồng)
- Gap >= 2.0s → nối bằng `". "` (có ngắt câu)

## 5. Output format

Giữ nguyên format Parquet hiện tại (3 cột: audio struct, transcription, file_name).

## 6. Kết quả kỳ vọng

| | Trước | Sau |
|---|---|---|
| Số segment | ~1.059 | ~1.000 |
| Short (<2s) | còn nhiều | 0 |
| Overlap thời gian | có | 0 |
| Mỗi segment | 1-3 câu lộn xộn | 1 câu chuẩn hoặc vài câu ngắn gộp lại |
