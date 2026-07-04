# Vietnamese Punctuation Restoration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add punctuation restoration to the pipeline so videos without VTT punctuation (live streams, unscripted speech) can still produce usable TTS segments.

**Architecture:** A new `punctuator.py` module using `underthesea.sent_tokenize()` to segment raw text into sentences, then heuristic-based punctuation insertion (`.` / `?` / `!`). Inserted as a conditional step between `clean_text` and `dedup` — only runs when merged segments lack ending punctuation.

**Tech Stack:** Python 3.11+, underthesea (pip package)

## Global Constraints

- Must preserve existing behavior for videos that already have proper punctuation
- `restore_punctuation(text: str) -> str` — pure function, no side effects
- ~50ms per 1000 words on CPU
- Only activate when heuristics detect missing punctuation

---

## File Structure

| File | Action |
|------|--------|
| `tts_pipeline/punctuator.py` | **Create** — `restore_punctuation()` + helpers |
| `tts_pipeline/cli.py` | **Modify** — add import + conditional step |
| `tests/test_punctuator.py` | **Create** — 5 unit tests |
| `requirements.txt` | **Modify** — add `underthesea` |

---

### Task 1: Create `punctuator.py` module

**Files:**
- Create: `tts_pipeline/punctuator.py`
- Test: `tests/test_punctuator.py` (Task 2)

**Interfaces:**
- Produces: `restore_punctuation(text: str) -> str`

- [ ] **Step 1: Install underthesea**

```bash
pip install underthesea
```

Verify: `python -c "from underthesea import sent_tokenize; print(sent_tokenize('chào bạn bạn tên là gì'))"`

Expected output: `['chào bạn', 'bạn tên là gì']`

- [ ] **Step 2: Write `punctuator.py`**

Create `D:\MinhTh_code\DataAudioProcess\tts_pipeline\punctuator.py`:

```python
"""Vietnamese punctuation restoration for raw caption text."""

import re
from underthesea import sent_tokenize

_QUESTION_WORDS = {
    "ai", "gì", "đâu", "sao", "nào", "bao nhiêu",
    "tại sao", "vì sao", "thế nào", "ra sao",
    "à", "hả", "nhỉ", "nhé", "chứ", "chăng", "ư", "hử", "hở",
}

_EXCLAMATION_STARTS = {
    "trời", "chao", "ôi", "á", "úi", "ối", "eo",
    "chà", "ồ", "ô hay", "ôi trời", "trời ơi", "trời đất",
}


def _detect_punctuation(sentence: str) -> str:
    """Detect end punctuation for a sentence.

    Returns one of '.', '?', or '!'.
    """
    lower = sentence.lower().strip()
    if not lower:
        return "."

    # Exclamation if sentence starts with an exclamation word
    for start_word in _EXCLAMATION_STARTS:
        if lower.startswith(start_word):
            return "!"

    # Question if sentence contains question words
    words = set(re.sub(r"[^\w\s]", " ", lower).split())
    if words & _QUESTION_WORDS:
        return "?"

    return "."


def restore_punctuation(text: str) -> str:
    """Add end punctuation (., ? , !) and capitalize sentences in raw text.

    Uses underthesea.sent_tokenize() for Vietnamese-aware sentence
    segmentation, then appends a period, question mark, or exclamation
    mark based on content heuristics.

    Args:
        text: Raw text without punctuation.

    Returns:
        Text with punctuation added and sentences capitalized.
    """
    if not text or not text.strip():
        return text

    sentences = sent_tokenize(text)
    if not sentences:
        return text

    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        punct = _detect_punctuation(sent)

        # Capitalize first letter
        first_char = sent[0]
        if first_char.islower():
            sent = first_char.upper() + sent[1:]

        result.append(sent + punct)

    return " ".join(result)
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from tts_pipeline.punctuator import restore_punctuation; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Quick manual test**

Run:
```python
PYTHONIOENCODING=utf-8 python -c "
from tts_pipeline.punctuator import restore_punctuation
tests = [
    ('alo Chào mừng các bạn', 'Alo. Chào mừng các bạn.'),
    ('bạn tên là gì', 'Bạn tên là gì?'),
    ('trời ơi sao đẹp thế', 'Trời ơi sao đẹp thế!'),
    ('Hello world.', 'Hello world.'),
]
for inp, exp in tests:
    out = restore_punctuation(inp)
    status = 'PASS' if out == exp else 'FAIL'
    print(f'{status}: \"{inp}\" → \"{out}\" (expected: \"{exp}\")')
"
```

- [ ] **Step 5: Commit**

```bash
git add tts_pipeline/punctuator.py
git commit -m "feat: add punctuator module for Vietnamese punctuation restoration

Uses underthesea.sent_tokenize() for sentence segmentation and
heuristic-based (. / ? / !) end punctuation detection.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add tests for `punctuator.py`

**Files:**
- Create: `tests/test_punctuator.py`

- [ ] **Step 1: Write `test_punctuator.py`**

Create `D:\MinhTh_code\DataAudioProcess\tests\test_punctuator.py`:

```python
"""Tests for Vietnamese punctuation restoration."""

from tts_pipeline.punctuator import restore_punctuation, _detect_punctuation


def test_restore_empty_text():
    assert restore_punctuation("") == ""
    assert restore_punctuation("   ") == "   "


def test_restore_already_punctuated():
    """Text already ending with punctuation should not double-punctuate."""
    result = restore_punctuation("Xin chào. Tôi là Bách.")
    assert result == "Xin chào. Tôi là Bách."


def test_restore_adds_period():
    """Raw text without punctuation gets periods added."""
    result = restore_punctuation("xin chào các bạn")
    assert result == "Xin chào các bạn."


def test_restore_question():
    """Question words trigger ? punctuation."""
    result = restore_punctuation("bạn tên là gì")
    assert result == "Bạn tên là gì?"


def test_restore_exclamation():
    """Exclamation-starting sentences get ! punctuation."""
    result = restore_punctuation("trời ơi sao đẹp thế")
    assert result == "Trời ơi sao đẹp thế!"


def test_restore_multiple_sentences():
    """Multiple sentences each receive proper punctuation."""
    result = restore_punctuation("chào bạn mình tên là bách bạn tên là gì trời ơi vui quá")
    # underthesea may segment as: ["chào bạn", "mình tên là bách", "bạn tên là gì", "trời ơi vui quá"]
    assert "?" in result
    assert "!" in result
    assert result[0].isupper()


def test_detect_period():
    assert _detect_punctuation("xin chào các bạn") == "."


def test_detect_question():
    assert _detect_punctuation("bạn tên là gì") == "?"


def test_detect_exclamation():
    assert _detect_punctuation("trời ơi sao đẹp thế") == "!"
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_punctuator.py -v`
Expected: 9/9 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_punctuator.py
git commit -m "test: add unit tests for punctuator module

Tests cover: empty text, already punctuated, period insertion,
question detection, exclamation detection, multiple sentences,
and individual _detect_punctuation cases.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Update `cli.py` — add punctuation step to pipeline

**Files:**
- Modify: `tts_pipeline/cli.py`

**Interfaces:**
- Consumes: `restore_punctuation` from `tts_pipeline.punctuator`

- [ ] **Step 1: Add import**

After `from tts_pipeline.processor import ...` add:

```python
from tts_pipeline.punctuator import restore_punctuation
```

- [ ] **Step 2: Add punctuation step after clean_text**

In `run_pipeline()`, after the `clean_text` loop and before `dedup`, add:

```python
    # Punctuation restoration (only if text lacks .!?)
    no_punct = [seg for seg in merged
                if seg["text"] and seg["text"][-1] not in ".!?"]
    if no_punct:
        for seg in merged:
            if seg["text"]:
                seg["text"] = restore_punctuation(seg["text"])
    print(f"   After punctuation: {len(merged)}")
```

The full pipeline block becomes:

```python
    for seg in merged:
        seg["text"] = clean_text(seg["text"])

    # Punctuation restoration (only if text lacks .!?)
    no_punct = [seg for seg in merged
                if seg["text"] and seg["text"][-1] not in ".!?"]
    if no_punct:
        for seg in merged:
            if seg["text"]:
                seg["text"] = restore_punctuation(seg["text"])
    print(f"   After punctuation: {len(merged)}")

    merged = dedup_consecutive_text(merged)
    print(f"   After dedup: {len(merged)}")
```

- [ ] **Step 3: Verify pipeline imports work**

Run: `python -m tts_pipeline.cli --help`
Expected: help message displayed, no import errors

- [ ] **Step 4: Verify existing tests still pass**

Run: `python -m pytest tests/ -v`
Expected: 33/33 passed (24 old + 9 new)

- [ ] **Step 5: Commit**

```bash
git add tts_pipeline/cli.py
git commit -m "feat: add punctuation restoration step to pipeline

Inserted between clean_text and dedup. Only activates when merged
segments lack sentence-ending punctuation (. ! ?).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add underthesea**

Add to `D:\MinhTh_code\DataAudioProcess\requirements.txt`:

```
underthesea>=6.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add underthesea dependency for punctuation restoration

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Run full pipeline and verify

- [ ] **Step 1: Test on video WITHOUT punctuation (T_zgDuLSIYU)**

```bash
rm -rf dataset_test/dataset dataset_test/audio dataset_test/*.parquet
python pipeline_tts.py "https://www.youtube.com/watch?v=T_zgDuLSIYU" --raw raw --output dataset_test 2>&1 | grep -vE "\[download\]|Cut audio" | grep -v "^$"
```

Expected output:
- After merge: ~133+ segments (with force-break)
- After punctuation: same count, but all segments now end with `.`, `?`, or `!`
- After segment_by_content: more segments than before (because punctuation enables better grouping)
- After filter: >100 segments

- [ ] **Step 2: Test on video WITH punctuation (iUGFXuxHNAA)**

```bash
rm -rf dataset_test/*
python pipeline_tts.py "https://www.youtube.com/watch?v=iUGFXuxHNAA" --raw raw --output dataset_test 2>&1 | grep -vE "\[download\]|Cut audio" | grep -v "^$"
```

Expected: pipeline should skip punctuation step (already has `. !?`) and produce same results as before (~127 segments).

- [ ] **Step 3: Verify output quality**

```bash
PYTHONIOENCODING=utf-8 python -c "
import pyarrow.parquet as pq
t = pq.read_table('dataset_test/T_zgDuLSIYU_train.parquet')
txts = t.column('transcription').to_pylist()
no_punct = [t for t in txts if t and t[-1] not in '.!?']
print(f'Total: {len(txts)}, Missing punctuation: {len(no_punct)}')
if no_punct:
    print(f'  Sample: \"{no_punct[0][:60]}\"')
else:
    print('  ALL segments have proper punctuation!')
"
```

- [ ] **Step 4: Commit final changes**

```bash
git add _analyze2.py  # if any pipeline changes need sync
git log --oneline -5
```
