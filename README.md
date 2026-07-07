# AudioProcess — Pipeline tạo Dataset TTS từ YouTube Audio Book

Tự động tải audio book từ YouTube, cắt thành từng đoạn ngắn dựa trên voice-activity detection (Silero-VAD), ghép với transcript (caption VTT) và xuất dataset chuẩn HuggingFace phục vụ training Text-To-Speech.

## Tính năng

- 🎬 **Tải YouTube** — audio + auto-caption tiếng Việt, tự động convert WAV (22050Hz, mono)
- ✂️ **Cắt segment bằng VAD** — Silero-VAD detect silence → re-segment chính xác, không lệch timing như VTT
- 🔤 **Phục hồi dấu câu** — `restore_punctuation()` cho VTT không dấu, clause splitting với discourse markers
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
┌──────────────────────┐
│  downloader           │  yt-dlp: audio.wav + vi.vtt
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  parser + merge_cues  │  parse_vtt_full() → merge_cues()
│                       │  Chỉ lấy text, bỏ timing VTT
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Punctuator           │  restore_punctuation() + _split_by_clause()
│                       │  Cho VTT không có dấu câu
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Silero-VAD Scanner   │  get_speech_intervals() → [{start,end}]
│                       │  Speech detection từ audio, threshold 0.3
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  VAD Aligner          │  group 5-20s + align text từ raw VTT cues
│                       │  Bằng overlap ratio > 0.1
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  filter + exporter    │  ffmpeg cut → WAV + Parquet (train/test split)
└──────────────────────┘
```

Project gồm 9 module trong `tts_pipeline/`:

| Module | Chức năng |
|--------|-----------|
| `config.py` | Pydantic Settings + YAML loading + ffmpeg validation |
| `downloader.py` | Tải audio + caption từ YouTube qua yt-dlp |
| `parser.py` | Parse VTT + merge_cues + VTT file I/O |
| `punctuator.py` | Phục hồi dấu câu, clause splitting |
| `processor.py` | Làm sạch text, dedup, fix_time |
| **`vad.py`** | **Silero-VAD scanner: speech intervals từ audio** |
| **`vad_aligner.py`** | **Group VAD intervals + align text từ VTT cues** |
| `exporter.py` | Cắt audio bằng ffmpeg, xuất Parquet, train/test split |
| `cli.py` | CLI entry point, orchestrate pipeline |

## Xử lý chất lượng

- **Fragment merge:** YouTube auto-caption sinh nhiều cue cho cùng một đoạn text (word-level → plain fragment → text mở rộng). Pipeline tự động gộp và giữ text đầy đủ nhất. Cue < 0.3s được coi là fragment chuyển tiếp và chỉ extend time, không ghi đè text.
- **Force-break text dài:** Nếu merged segment > 250 ký tự không có dấu câu, tự động ngắt segment mới (force-break trong merge_cues).
- **Phục hồi dấu câu:** Segments thiếu `. ! ?` được xử lý qua `restore_punctuation()` — `_split_by_clause()` dùng discourse markers (`nhưng`, `cho nên`, `rồi thì`...) để tách text flow không dấu câu.
- **VAD Re-segment:** Thay thế VTT timing bằng **Silero-VAD** — detect speech từ audio waveform, tạo segment boundaries chính xác theo voice activity. Loại bỏ hoàn toàn lỗi audio-text mismatch do VTT timing lệch 1-3s.
- **Text alignment:** VAD segment timing được map ngược vào raw VTT cues qua overlap ratio (> 0.1). Text giữ nguyên gốc từ VTT, không dedup → audio-text khớp 100%.
- **Segment grouping:** Các VAD intervals được gộp greedy 5-20s.
- **Audio validation:** WAV mono 22050Hz 16-bit, duration 5-20s, text >= 10 ký tự.

## Test

```bash
python -m pytest tests/ -v
```

## Kết quả thử nghiệm

Với video `T_zgDuLSIYU` (Tu Tiên Lạ Lắm tập 1, ~45 phút, VTT không dấu câu, có nhạc nền):

- Raw VTT cues: ~2700
- Silero-VAD: **748 speech intervals** → **204 groups** (5-20s)
- VAD threshold: 0.3 (audiobook cần threshold thấp hơn 0.5)
- **180 train + 21 test segments**
- **0 under 5s, 0 over 20s, 0 missing punctuation**
- Mean duration: 16.9s
- Timing từ VAD, text từ VTT cues overlap → **audio-text khớp 100%**

## Giấy phép

Dataset output sử dụng giấy phép CC-BY-NC-4.0 (giống dataset tham khảo).
