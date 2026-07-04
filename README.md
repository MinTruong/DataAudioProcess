# AudioProcess — Pipeline tạo Dataset TTS từ YouTube Audio Book

Tự động tải audio book từ YouTube, cắt thành từng đoạn ngắn, ghép với transcript (caption) và xuất dataset chuẩn HuggingFace phục vụ training Text-To-Speech.

## Tính năng

- 🎬 **Tải YouTube** — audio + auto-caption tiếng Việt, tự động convert WAV (22050Hz, mono)
- ✂️ **Cắt segment thông minh** — gộp fragment YouTube, xoá overlap, phục hồi dấu câu, tách clause, lọc đuôi lặp
- 📊 **Xuất dataset chuẩn** — Parquet + WAV files, format tương thích HuggingFace datasets
- 🔀 **Train/test split** — mặc định 90/10
- ⚙️ **Config bằng YAML** — không cần sửa code
- 📦 **Batch processing** — xử lý nhiều video cùng lúc
- ☁️ **Upload HuggingFace** — đẩy dataset lên HF Hub bằng một lệnh

## Yêu cầu

- **Python** 3.12+
- **ffmpeg** (phải có trên PATH — kiểm tra bằng `ffmpeg -version`)
- **pip** packages (xem `requirements.txt`)

## Cài đặt nhanh

```bash
# Clone repo
git clone <repo-url> && cd AudioProcess

# Cài dependencies
pip install -r requirements.txt
```

## Sử dụng

### 1. Pipeline cơ bản — 1 video

```bash
python pipeline_tts.py "https://www.youtube.com/watch?v=..."
```

Output: `dataset/{video_id}_train.parquet` + `dataset/{video_id}_test.parquet` + WAV files trong `dataset/audio/`.

### 2. Channel — lấy N video đầu

```bash
python pipeline_tts.py "https://www.youtube.com/@backaudio" --max-videos 3
```

### 3. Tuỳ chỉnh output

```bash
python pipeline_tts.py <url> --output my_dataset --raw my_raw
```

### 4. Config bằng YAML

Tạo file `configs/my_config.yaml`:

```yaml
sample_rate: 22050
min_segment_dur: 2.0
max_segment_dur: 20.0
min_text_len: 10
max_text_len: 500
train_split: 0.9
```

Sau đó chạy:

```bash
python pipeline_tts.py <url> --config configs/my_config.yaml
```

### 5. Batch — xử lý nhiều video

Tạo file `urls.txt` (mỗi dòng một URL), sau đó:

```bash
python scripts/batch_process.py --file urls.txt
```

Kết quả: batch summary JSON tại `dataset/batch_summary.json`.

### 6. Upload lên HuggingFace Hub

Cách 1 — dùng file config:

Tạo `configs/upload.yaml`:

```yaml
repo: your-username/your-dataset-name
dir: dataset
language: vi
private: false
commit_message: "Add TTS dataset from audio books"
```

Chạy:

```bash
python scripts/push_to_hub.py --config configs/upload.yaml
```

> **Token HF:** có thể set trong config YAML hoặc dùng `huggingface-cli login`. Nếu chưa có token, vào [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) tạo một cái.

Cách 2 — CLI trực tiếp:

```bash
python scripts/push_to_hub.py --dir dataset/ --repo your-username/dataset-name
```

CLI args ghi đè config YAML:

```bash
python scripts/push_to_hub.py --config configs/upload.yaml --private
```

## Output format

```
dataset/
├── {video_id}_train.parquet    ← 90% dữ liệu
├── {video_id}_test.parquet     ← 10% dữ liệu
└── audio/
    ├── {video_id}_00000.wav
    ├── {video_id}_00001.wav
    └── ...
```

Mỗi file Parquet có các cột:

| Column | Type | Description |
|--------|------|-------------|
| `audio` | struct | `{path: string, bytes: binary}` — path relative + audio bytes |
| `transcription` | string | Nội dung transcript tiếng Việt |
| `file_name` | string | Tên file WAV (`audio/xxx.wav`) |

Thông số kỹ thuật audio: WAV, mono, 22050Hz, 16-bit signed integer PCM.

## Kiến trúc

```
YouTube URL
    │
    ▼
┌──────────────┐
│  downloader  │  yt-dlp: audio + VTT caption (tiếng Việt)
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│  parser + merge   │  parse_vtt_full() → merge_cues() → force-break text dài
│  (export _merged  │  Xuất VTT file để inspect
│   .vtt)           │
└──────┬────────────┘
       │
       ▼
┌────────────────────────────┐
│  punctuator (qua VTT file) │  restore_punctuation() + _split_by_clause()
│                            │  Cho segments thiếu dấu câu .
│                            │  ! ? → split + prorate time nội bộ
│                            │  Lọc đuôi ngắn lặp (VTT overlap artifact)
└──────┬─────────────────────┘
       │
       ▼
┌──────────────────┐
│ dedup_consecutive │  Xoá overlap text giữa segment kế
│ _text             │  (suffix ở MỌI vị trí, max 15 từ)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ fix_time_overlaps │  Đẩy start time nếu 2 segment chồng lấn
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ segment_by_       │  NHÓM atomic segments thành group 5-20s
│ content           │  (greedy, boundary trim 0.15s+0.3s)
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  exporter    │  ffmpeg cut → WAV + Parquet (train/test split)
└──────────────┘
```

Project gồm 7 module trong `tts_pipeline/`:

| Module | Chức năng |
|--------|-----------|
| `config.py` | Pydantic Settings + YAML loading + ffmpeg validation |
| `downloader.py` | Tải audio + caption từ YouTube qua yt-dlp |
| `parser.py` | Parse VTT + merge_cues + VTT file I/O |
| `punctuator.py` | Phục hồi dấu câu, clause splitting, lọc đuôi lặp |
| `processor.py` | Làm sạch text, dedup, fix_time, segment_by_content |
| `exporter.py` | Cắt audio bằng ffmpeg, xuất Parquet, train/test split |
| `cli.py` | CLI entry point, orchestrate pipeline |

## Xử lý chất lượng

- **Fragment merge:** YouTube auto-caption sinh nhiều cue cho cùng một đoạn text (word-level → plain fragment → text mở rộng). Pipeline tự động gộp và giữ text đầy đủ nhất. Cue < 0.3s được coi là fragment chuyển tiếp và chỉ extend time, không ghi đè text.
- **Force-break text dài:** Nếu merged segment > 250 ký tự không có dấu câu, tự động ngắt segment mới (force-break trong merge_cues).
- **Phục hồi dấu câu:** Segments thiếu `. ! ?` được xử lý qua pipeline VTT file — `_split_by_clause()` dùng discourse markers (`nhưng`, `cho nên`, `rồi thì`...) để tách text flow không dấu câu, phục hồi punctuation bằng heuristics (từ hỏi → ?, cảm thán → !, còn lại → .).
- **Lọc đuôi lặp:** Segment <= 3 từ trùng với đuôi segment trước (VTT overlap artifact) bị loại bỏ.
- **Overlap dedup:** Đoạn kế tiếp thường lặp lại 1-15 từ cuối của đoạn trước (`dedup_consecutive_text` tự động cắt bỏ).
- **Segment grouping:** Các merged segments được nhóm greedy 5-20s, mỗi segment là atomic unit (không split nội bộ để tránh audio-text mismatch từ character-ratio proration).
- **Boundary trim:** 0.15s đầu + 0.3s cuối mỗi group để tránh VTT overlap artifact.
- **Audio validation:** WAV mono 22050Hz 16-bit, duration 5-20s, text >= 10 ký tự.

## Test

```bash
python -m pytest tests/ -v
```

## Kết quả thử nghiệm

Với video `iUGFXuxHNAA` (~42 phút):

- Raw VTT cues: ~2700
- Sau merge_cues: 174 segments
- Sau punctuation + dedup + segment_by_content: **198 segments**
- **0 segments under 5s, 0 over 20s, 0 missing punctuation**
- Mean duration: 15.7s

Với video `T_zgDuLSIYU` (Tu Tiên Lạ Lắm tập 1, ~45 phút, VTT không dấu câu):

- Raw VTT cues: ~2700
- Sau merge_cues: 217 segments
- Sau punctuation (clause split) + dedup + segment_by_content: **222 segments**
- **0 segments under 5s, 0 over 20s, 0 missing punctuation**
- Mean duration: 13.9s

## Giấy phép

Dataset output sử dụng giấy phép CC-BY-NC-4.0 (giống dataset tham khảo).
