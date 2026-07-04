# SPEC: Vietnamese Punctuation Restoration for TTS Pipeline

**Date:** 2026-07-04
**Status:** Draft

## Vấn đề

YouTube auto-caption cho một số video (đặc biệt là live stream, trò chuyện) không có dấu câu `. , ! ?`. Ví dụ:

```
alo Chào mừng các cầu thủ có tiếng rồi đó các bạn ơi nãy giờ là Bách quên là kết nối với lại cái micro
```

Kết quả: `merge_cues` không thể split thành segments có ý → 2741 cues chỉ thành 1 segment duy nhất → bị filter → mất hết dữ liệu.

## Giải pháp

Thêm một bước **punctuation restoration** vào pipeline: dùng `underthesea` (thư viện NLP tiếng Việt nhẹ, ~50MB) để đặt lại dấu câu cho text raw.

## Architecture

```python
pipeline_order = merge → clean → punctuate → dedup → fix_time → segment_by_content → fix_time → filter → export
```

Module mới: `tts_pipeline/punctuator.py`

### Bước 1: Sentence segmentation

Dùng `underthesea.sent_tokenize()` để tách text thành câu:

```python
from underthesea import sent_tokenize

sentences = sent_tokenize("alo chào mừng các bạn nãy giờ là bách quên là kết nối micro")
# → ["alo", "chào mừng các bạn", "nãy giờ là bách quên là kết nối micro"]
```

### Bước 2: Punctuation insertion

Mỗi câu được gán dấu `.` (dấu chấm) ở cuối. Quy tắc bổ sung:

- Nếu câu chứa từ để hỏi (`ai`, `gì`, `đâu`, `sao`, `không?`, `à?`, `hử?`, `nhỉ?`, `chứ?`) và text ngắn (< 100 từ) → `?`
- Nếu câu bắt đầu bằng từ cảm thán (`chao ôi`, `trời ơi`, `ôi`) → `!`
- Mặc định → `.`

### Bước 3: Hậu xử lý

- Viết hoa chữ cái đầu mỗi câu
- Nối lại thành text hoàn chỉnh

```python
"alo. Chào mừng các bạn. Nãy giờ là bách quên là kết nối cái micro."
```

## Việc cần làm

1. **Module `tts_pipeline/punctuator.py`** — hàm `restore_punctuation(text: str) -> str`
   - `sent_tokenize` → list câu
   - `_add_end_punctuation(sentences)` → [".", "?", "!"]
   - `_capitalize(sentences)` → viết hoa
   - `_join(sentences, punctuations)` → text hoàn chỉnh

2. **Pipeline update** — thêm bước `punctuate` vào `cli.py`

3. **Test** — `tests/test_punctuator.py`
   - test với text đã có dấu câu → không thay đổi
   - test với text không dấu → thêm dấu
   - test câu hỏi → `?`
   - test câu cảm thán → `!`
   - test câu thường → `.`

4. **Dependency** — thêm `underthesea` vào `requirements.txt`

## Yêu cầu kỹ thuật

- `underthesea` bản mới nhất: `pip install underthesea`
- Chạy trên CPU, inference ~50ms cho 1000 từ
- Không cần GPU, không cần download model nặng
- File ~50MB cho dictionary + model mặc định

## Input/Output

| Input | Output |
|-------|--------|
| `"alo Chào mừng các bạn"` | `"Alo. Chào mừng các bạn."` |
| `"bạn tên là gì"` | `"Bạn tên là gì?"` |
| `"trời ơi sao đẹp thế"` | `"Trời ơi sao đẹp thế!"` |

## Kiến trúc chi tiết

### `punctuator.py`

```python
import re
from underthesea import sent_tokenize

_QUESTION_WORDS = {"ai", "gì", "đâu", "sao", "nào", "đâu", "bao nhiêu",
                   "tại sao", "vì sao", "thế nào", "ra sao", "à", "hả",
                   "nhỉ", "nhé", "chứ", "chăng", "ư", "hử", "hở", "nhỉ?"}

_EXCLAMATION_STARTS = {"trời", "chao", "ôi", "á", "úi", "ối", "eo",
                       "chà", "ồ", "ô hay", "ôi trời", "trời ơi"}

def restore_punctuation(text: str) -> str:
    # 1. sent_tokenize handles basic segmentation
    sentences = sent_tokenize(text)
    if not sentences:
        return text

    # 2. Add end punctuation
    punctuated = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        punct = _detect_punctuation(sent)
        # Capitalize first letter
        if sent[0].islower():
            sent = sent[0].upper() + sent[1:]
        punctuated.append(sent + punct)

    return " ".join(punctuated)

def _detect_punctuation(sentence: str) -> str:
    lower = sentence.lower().strip()
    if any(lower.startswith(w) for w in _EXCLAMATION_STARTS):
        return "!"
    words = set(re.sub(r"[^\w\s]", " ", lower).split())
    if words & _QUESTION_WORDS:
        return "?"
    return "."
```

## Pipeline change

`cli.py`: thêm `from tts_pipeline.punctuator import restore_punctuation`

```python
# After clean_text
merged = [seg for seg in merged if seg["text"]]
if any(not seg["text"][-1].isascii() or seg["text"][-1] not in ".!?" for seg in merged):
    for seg in merged:
        seg["text"] = restore_punctuation(seg["text"])
```

Chỉ chạy punctuator nếu text thiếu dấu câu (heuristic) để tránh tốn thời gian cho video đã có caption chất lượng.

## Verification

1. `python -m pytest tests/test_punctuator.py -v` — 5 tests pass
2. Test trên video raw:
   ```python
   from tts_pipeline.punctuator import restore_punctuation
   from tts_pipeline.parser import parse_vtt_full, merge_cues
   raw = parse_vtt_full('raw/T_zgDuLSIYU.vi.vtt')
   merged = merge_cues(raw)
   # Check if segments have ending punctuation
   broken = [s for s in merged if s["text"] and s["text"][-1] not in ".!?"]
   print(f"{len(broken)}/{len(merged)} segments without punctuation")
   ```
3. Chạy full pipeline: `python pipeline_tts.py https://youtu.be/T_zgDuLSIYU`

## Quyết định thiết kế

- **Dùng `underthesea` thay vì model BERT** vì: nhẹ (~50MB), không cần GPU, không cần fine-tune, chạy CPU trong 50ms/1000 từ, tích hợp bằng `pip install` đơn giản
- **Không dùng regex thuần** vì: underthesea có CRF-based sentence segmentation hiểu được ngữ cảnh tiếng Việt tốt hơn regex
- **Chỉ chạy khi cần** vì: hầu hết video đã có dấu câu từ YouTube VTT, không cần punctuation restoration
