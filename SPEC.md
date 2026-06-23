# SPEC: Pipeline tạo TTS Dataset từ YouTube Audio Book

## 1. Tổng quan

**Mục tiêu:** Xây dựng pipeline tự động tạo dataset TTS (Text-To-Speech) từ nguồn video YouTube audio book, xuất ra format tương thích với HuggingFace datasets.

**Đầu vào:** URL video YouTube (hoặc channel/playlist) có auto-caption tiếng Việt.
**Đầu ra:** Dataset Parquet + WAV files, format:
| Column | Type | Ghi chú |
|--------|------|---------|
| `audio` | struct | `{path: string, bytes: binary}` — path relative + audio bytes |
| `transcription` | string | 10-500 ký tự tiếng Việt |
| `file_name` | string | Tên file WAV (`audio/xxx.wav`) |

**Yêu cầu hệ thống:** Python 3.12+, ffmpeg, pip packages: yt-dlp, pandas, pyarrow, tqdm.

---

## 2. Kiến trúc Pipeline

```
YouTube URL
    │
    ▼
┌─────────────────────┐
│  Bước 1: Download   │  yt-dlp: audio (WAV 22050Hz) + auto-caption (VTT)
│  video + caption    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Bước 2: Parse VTT  │  Parse VTT → raw cues → merge fragments → dedup overlap
│  & Merge fragments   │  → split long segments → clean text
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Bước 3: Cut audio  │  ffmpeg: cắt từng segment theo [start, end]
│  segments           │  WAV mono 22050Hz 16-bit
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Bước 4: Export     │  pandas DataFrame → Parquet file
│  Parquet dataset    │  + WAV files trong audio/
└─────────────────────┘
```

---

## 3. Luồng xử lý chi tiết

### 3.1. Bước 1: Download

**Công cụ:** yt-dlp

**Tham số:**
- Format: `bestaudio/best`
- Post-process: Extract WAV, mono, 22050Hz
- Subtitle: `writeautomaticsub` — tải auto-caption tiếng Việt
- Output: `raw/{video_id}.wav` + `raw/{video_id}.vi.vtt`

**Xử lý lỗi:**
- Video không có caption tiếng Việt → báo lỗi, skip
- Audio không tải được → fallback convert từ format khác (mp3/m4a/opus)

### 3.2. Bước 2: Parse VTT

#### 3.2.1. Raw cues

YouTube auto-caption VTT có cấu trúc đặc thù:

```
00:00:01.560 --> 00:00:03.350 align:start position:0%
<v ...>Chào mừng tất cả các đạo hữu đã quay trở

00:00:03.350 --> 00:00:03.360 align:start position:0%
Chào mừng tất cả các đạo hữu đã quay trở           ← fragment (0.01s)

00:00:03.360 --> 00:00:06.190 align:start position:0%
Chào mừng tất cả các đạo hữu đã quay trở
lại<00:00:03.520><c> với</c><00:00:03.639><c> kênh</c>... ← word-level timestamps
```

**Thuật toán:**
1. Duyệt từng dòng, phát hiện timestamp (`-->`)
2. Gom text giữa các timestamp
3. Strip VTT markup: `<c>...</c>`, `<v>...</v>`, `<timestamp>`

#### 3.2.2. Merge fragments

YouTube auto-caption có 3 cue cho cùng 1 đoạn text:
- Cue A: word-level markup (dài)
- Cue B: plain text, duration 0.01s (fragment chuyển tiếp)
- Cue C: text mở rộng (của cue kế tiếp)

**Logic:**
- Fragment có duration < 0.3s → bỏ qua text, kéo dài end time của segment trước
- Cue có text dài hơn và bắt đầu bằng text của cue trước → gộp (accumulate text)
- Gap > 0.5s → không gộp (bắt đầu câu mới)

#### 3.2.3. Dedup overlap

Sau merge, các segment kế nhau thường bị overlap (cùng 1-3 từ cuối của segment N xuất hiện ở đầu segment N+1).

**Logic:**
1. Lấy 3 từ cuối của segment N
2. Nếu segment N+1 bắt đầu bằng cụm đó → cắt bỏ
3. Kiểm tra thêm 1 từ cuối nếu chưa đủ

#### 3.2.4. Split long segments

Segment > 20s → chia nhỏ bằng dấu câu (`.` trước, fallback `,`).
Thời gian phân bố đều: `char_ratio = total_dur / total_chars`.

#### 3.2.5. Text cleaning

```python
re.sub(r'\s+', ' ', text).strip()
# Giữ nguyên ký tự tiếng Việt (unicode)
```

### 3.3. Bước 3: Cut audio

**Công cụ:** ffmpeg

```bash
ffmpeg -y -ss {start} -t {duration} -i input.wav \
       -ar 22050 -ac 1 -sample_fmt s16 output.wav
```

**Validation:**
- Duration: 2.0s ≤ t ≤ 20.0s
- Text length: 10 ≤ chars ≤ 500

### 3.4. Bước 4: Export

**Format:** Parquet (Apache Arrow)

**Cấu trúc thư mục:**
```
dataset/{video_id}/
├── {video_id}_dataset.parquet
└── audio/
    ├── {video_id}_00000.wav
    ├── {video_id}_00001.wav
    └── ...
```

### 3.5. Train/test split

Khi export, dataset được chia train/test theo tỉ lệ `train_split` (default 0.9) trong config.
Output gồm 2 file Parquet riêng: `{video_id}_train.parquet` và `{video_id}_test.parquet`.
Có thể disable split bằng `split_train_test=False` khi gọi `export_dataset()`.

### 3.6. Config hệ thống

Pipeline dùng Pydantic Settings, có thể nạp từ file YAML:
```yaml
# configs/default.yaml
sample_rate: 22050
min_segment_dur: 2.0
max_segment_dur: 20.0
train_split: 0.9
```

Ghi đè qua CLI: `--output dir --raw dir --config path/to/config.yaml`

---

## 4. Kết quả thử nghiệm

### Môi trường
- OS: Windows 11
- Python 3.12.9
- ffmpeg 8.1.1
- yt-dlp 2026.6.9

### Video test: Tần Lão Yêu Tinh tập 640 (iUGFXuxHNAA)
- Duration: ~42 phút
- Raw segments (VTT cues): ~4.000
- Sau merge: ~1.200
- Sau dedup: ~1.100
- **Xuất được: 1.059 segments**
- **Tổng thời lượng: 44.8 phút**
- **Dung lượng WAV: 113 MB**

### Sample dataset (100 segments đầu)
- `sample_dataset/sample_dataset.parquet`
- 100 files WAV trong `sample_dataset/audio/`
- 4.5 phút, 11.5 MB
- Chất lượng text: sạch, đọc được nội dung truyện

---

## 5. Hạn chế & Cải thiện

### Hiện tại
- Chỉ xử lý 1 video / 1 channel giới hạn số video
- Overlap còn tồn đọng nhẹ (trường hợp overlap 1-2 từ)
- Không có bước validate audio (kiểm tra audio bị rỗng / noise)

### Có thể cải thiện
- [ ] Voice activity detection (VAD) — cắt chính xác hơn theo voice, không theo caption
- [ ] Whisper transcription — fallback khi không có caption
- [ ] Audio augmentation (thêm noise, thay đổi pitch) cho TTS robust
- [ ] Train/test split
- [ ] Validation: kiểm tra CER/WER giữa caption và Whisper transcription
- [ ] UI đơn giản (gradio / streamlit)

---

## 6. Cách sử dụng

```bash
# 1 video
python pipeline_tts.py "https://youtube.com/watch?v=..."

# Channel - N video đầu
python pipeline_tts.py "https://youtube.com/@backaudio" --max-videos 3

# Tự chọn output
python pipeline_tts.py <url> --output my_dataset

# Debug: giữ raw files
# (mặc định raw/ được giữ lại)
```

### Output
```
raw/{video_id}.wav          ← Audio gốc
raw/{video_id}.vi.vtt       ← Caption gốc
dataset/{video_id}_dataset.parquet  ← Dataset Parquet
dataset/audio/{video_id}_*.wav      ← Audio segments
```
