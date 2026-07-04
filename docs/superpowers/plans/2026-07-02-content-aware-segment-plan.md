# Content-Aware Sentence Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `split_by_sentence` + `merge_short_segments` with a unified `segment_by_content()` function that groups atomic sentences into 5–20s segments with complete semantic units.

**Architecture:** A single function in `processor.py` extracts atomic sentences from merged segments (prorating time by character ratio), applies music/noise filtering, then greedily groups them by duration. `dedup` and `fix_time_overlaps` move before this function; a second `fix_time_overlaps` pass runs after.

**Tech Stack:** Python 3.11+, re (regex), pytest

## Global Constraints

- WAV output: mono, 22050Hz, 16-bit
- Pipeline must handle Vietnamese text with mid-sentence proper nouns (Tần Tang, La Lạc Ma Quân)
- Segment duration target: 5–20s (soft), hard filter: <2s removed
- ffmpeg must be on PATH (validated at startup)
- All existing tests must still pass (imports of deleted functions removed from test imports)

---

## File Structure

| File | Change |
|------|--------|
| `tts_pipeline/processor.py` | Add `segment_by_content()`, `_has_music_noise()`, enhance `dedup_consecutive_text()` to iterate all match positions. Keep `split_by_sentence` and `merge_short_segments` for reference but they are no longer used by pipeline. |
| `tts_pipeline/cli.py` | Update imports (remove `split_by_sentence`, `merge_short_segments`; add `segment_by_content`). Update pipeline order. |
| `tests/test_processor.py` | Add 7 new tests for `segment_by_content`, 1 new test for enhanced dedup, 1 integration test. Remove obsolete `test_split_by_sentence_*` and `test_merge_short_*` tests. |
| `_analyze2.py` | Update pipeline chain to match new pipeline order. |

---

### Task 1: Add `segment_by_content()` to processor.py

**Files:**
- Modify: `tts_pipeline/processor.py` — add helper + main function
- Test: `tests/test_processor.py` (covered in Task 3)

**Interfaces:**
- Consumes: `clean_text()` (existing) in same module
- Produces: `segment_by_content(segments: list[dict], min_dur: float = 5.0, max_dur: float = 20.0) -> list[dict]`

- [ ] **Step 1: Add `_has_music_noise()` helper**

Add to `processor.py` after `clean_text()`:

```python
_MUSIC_PATTERNS = ["[âm nhạc]", "[nhạc]", "[music]", "&gt;"]

def _has_music_noise(text: str) -> bool:
    """Check if text contains music/noise/copyright artifacts to filter out."""
    lower = text.lower().replace(" ", "")
    for p in _MUSIC_PATTERNS:
        if p in lower:
            return True
    return False
```

- [ ] **Step 2: Add `segment_by_content()` with Phase 1 + Phase 2**

Add after `_has_music_noise()`:

```python
def segment_by_content(
    segments: list[dict],
    min_dur: float = 5.0,
    max_dur: float = 20.0,
) -> list[dict]:
    """Group atomic sentences into 5–20s segments with complete semantic units.

    Phase 1 — Split each segment into atomic sentences by punctuation, prorate
    time by character ratio. Filter out music/noise artifacts.

    Phase 2 — Greedy left-to-right grouping:
    - If group duration < min_dur: append candidate unconditionally
    - If adding candidate would exceed max_dur: emit current group, start new
    - Otherwise: append candidate
    Tail group is always emitted (no backward merging into previous group).
    """
    # Phase 1: Extract atomic sentences with prorated time
    atoms: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg["text"]
        if not text.strip():
            continue

        parts = re.split(r"(?<=[.!?])\s+", text)
        total_chars = sum(len(p) for p in parts)
        if total_chars == 0:
            continue

        char_ratio = dur / total_chars
        cur_start = seg["start"]

        for part in parts:
            part = part.strip()
            if not part or _has_music_noise(part):
                continue
            part_dur = len(part) * char_ratio
            atoms.append({
                "start": cur_start,
                "end": cur_start + part_dur,
                "text": part,
            })
            cur_start += part_dur

    # Phase 2: Greedy duration-based grouping
    if not atoms:
        return []

    groups: list[list[dict]] = []
    current = [atoms[0]]

    for atom in atoms[1:]:
        group_dur = current[-1]["end"] - current[0]["start"]
        add_dur = atom["end"] - atom["start"]

        if group_dur < min_dur:
            current.append(atom)          # must append — no orphan
        elif group_dur + add_dur > max_dur:
            groups.append(current)         # emit, start fresh
            current = [atom]
        else:
            current.append(atom)           # room to grow

    groups.append(current)                 # always emit tail

    output = []
    for group in groups:
        joined = " ".join(a["text"] for a in group)
        cleaned = clean_text(joined)
        output.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": cleaned,
        })

    return output
```

- [ ] **Step 3: Commit**

```bash
git add tts_pipeline/processor.py
git commit -m "feat: add _has_music_noise() and segment_by_content()

Phase 1 extracts atomic sentences with time proration + music filter.
Phase 2 greedily groups into 5-20s segments with complete semantic units.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Enhance `dedup_consecutive_text()` — iterate all match positions

**Files:**
- Modify: `tts_pipeline/processor.py` (function `dedup_consecutive_text`, lines 11-48)
- Test: `tests/test_processor.py` (added in Task 4)

**Interfaces:**
- Consumes: `dedup_consecutive_text(segments) -> list[dict]` — existing signature unchanged
- Produces: same signature, now iterates all match positions per suffix length

**Problem:** The current code uses `str.find()` which returns the FIRST match. If the suffix appears at a position where the preceding char fails the word-boundary check, it drops to a shorter suffix immediately — even if the same suffix appears later at a valid boundary.

**Fix:** For each suffix length (from 15 down to 1), find ALL positions where the suffix appears in curr text. For each valid position (boundary check passes), accept. Pick the first valid match (smallest position = strips least text).

- [ ] **Step 1: Modify the overlap-finding loop**

Change lines 29-43 in `processor.py`:

**Old code:**
```python
            limit = min(len(prev_words), 15)
            for n in range(limit, 0, -1):
                prev_suffix = " ".join(prev_words[-n:])
                pos = curr_lower.find(prev_suffix.lower())
                if pos == -1:
                    continue
                if pos > 0 and not curr_lower[pos - 1].isspace():
                    continue
                text = text[pos + len(prev_suffix):].strip()
                text = text.lstrip(",").strip()
                if not text:
                    continue
                text = text[0].upper() + text[1:]
                break
```

**New code:**
```python
            limit = min(len(prev_words), 15)
            for n in range(limit, 0, -1):
                prev_suffix = " ".join(prev_words[-n:])
                search_start = 0
                while True:
                    pos = curr_lower.find(prev_suffix.lower(), search_start)
                    if pos == -1:
                        break
                    if pos == 0 or curr_lower[pos - 1].isspace():
                        # Valid word-boundary match — use it
                        text = text[pos + len(prev_suffix):].strip()
                        text = text.lstrip(",").strip()
                        if not text:
                            break  # empty result, try shorter suffix
                        text = text[0].upper() + text[1:]
                        break
                    search_start = pos + 1
                else:
                    continue  # no valid match at this n level
                break  # valid match found and processed
```

- [ ] **Step 2: Commit**

```bash
git add tts_pipeline/processor.py
git commit -m "fix: dedup_consecutive_text iterates all suffix match positions

Previously .find() returned only the first match. If that position failed
the word-boundary check, the code skipped to a shorter suffix even when
the same suffix appeared later at a valid boundary. Now iterates all
positions per suffix length.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Update `cli.py` — new pipeline order + imports

**Files:**
- Modify: `tts_pipeline/cli.py`

**Interfaces:**
- Consumes: `segment_by_content` from `tts_pipeline.processor`
- Consumes: `Settings.max_segment_dur` from config (already available)

- [ ] **Step 1: Update imports in cli.py**

Replace:
```python
from tts_pipeline.processor import (
    clean_text,
    dedup_consecutive_text,
    split_by_sentence,
    fix_time_overlaps,
    merge_short_segments,
)
```

With:
```python
from tts_pipeline.processor import (
    clean_text,
    dedup_consecutive_text,
    fix_time_overlaps,
    segment_by_content,
)
```

- [ ] **Step 2: Update pipeline processing block in `run_pipeline()`**

Replace lines 49-57:
```python
    for seg in merged:
        seg["text"] = clean_text(seg["text"])

    merged = split_by_sentence(merged, s.min_segment_dur)
    print(f"   After sentence split: {len(merged)}")

    merged = dedup_consecutive_text(merged)
    print(f"   After dedup: {len(merged)}")

    merged = fix_time_overlaps(merged)
    merged = merge_short_segments(merged, s.min_segment_dur, s.min_text_len)
    print(f"   After merge short: {len(merged)}")
```

With:
```python
    for seg in merged:
        seg["text"] = clean_text(seg["text"])

    merged = dedup_consecutive_text(merged)
    print(f"   After dedup: {len(merged)}")

    merged = fix_time_overlaps(merged)
    print(f"   After fix_time: {len(merged)}")

    merged = segment_by_content(merged, s.min_segment_dur, s.max_segment_dur)
    print(f"   After segment_by_content: {len(merged)}")

    merged = fix_time_overlaps(merged)
    print(f"   After fix_time (2nd pass): {len(merged)}")
```

- [ ] **Step 3: Run existing tests to verify cli imports still work**

Run: `python -m pytest tests/ -v`

Expected: import errors for `split_by_sentence` and `merge_short_segments` from test file (since tests still import them). That's expected — Task 4 will fix imports in test file.

- [ ] **Step 4: Verify CLI --help works**

Run: `python -m tts_pipeline.cli --help`

Expected: help message printed with no errors

- [ ] **Step 5: Commit**

```bash
git add tts_pipeline/cli.py
git commit -m "refactor: update cli.py pipeline order

Remove split_by_sentence and merge_short_segments.
New order: merge → clean → dedup → fix_time → segment_by_content
→ fix_time (2nd pass) → filter → export

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update tests — add segment_by_content tests, remove obsolete tests

**Files:**
- Modify: `tests/test_processor.py`

**Changes:**
- Remove imports of `split_by_sentence` and `merge_short_segments`
- Remove test cases: `test_split_by_sentence_short_unchanged`, `test_split_by_sentence_splits`, `test_merge_short_merges_into_next`, `test_merge_short_keeps_normal`
- Add 7 new tests for `segment_by_content`: basic grouping, overflow, single long, tail short, music filtered, orphan first
- Add 1 enhanced dedup test
- Add 1 integration test for the full pipeline chain

- [ ] **Step 1: Update imports**

Replace:
```python
from tts_pipeline.processor import (
    clean_text,
    dedup_consecutive_text,
    split_by_sentence,
    fix_time_overlaps,
    merge_short_segments,
)
```

With:
```python
from tts_pipeline.processor import (
    clean_text,
    dedup_consecutive_text,
    fix_time_overlaps,
    segment_by_content,
)
```

- [ ] **Step 2: Remove obsolete tests**

Delete these test functions:
```
test_split_by_sentence_short_unchanged
test_split_by_sentence_splits
test_merge_short_merges_into_next
test_merge_short_keeps_normal
```

- [ ] **Step 3: Add 7 segment_by_content tests**

```python
def test_segment_grouping_basic():
    """3 atomic sentences within min/max → 1 group."""
    segs = [{"start": 0, "end": 6, "text": "A. B. C."}]
    result = segment_by_content(segs, min_dur=5.0, max_dur=20.0)
    assert len(result) == 1
    assert result[0]["text"] == "A. B. C."
    assert abs(result[0]["end"] - result[0]["start"] - 6.0) < 0.01


def test_segment_grouping_overflow():
    """Many sentences totalling >20s → multiple groups (none exceeds max_dur)."""
    # 10 sentences "A. B. C. ... J." in 30s → each sentence ~3s
    text = ". ".join(chr(65 + i) for i in range(10)) + "."
    # text = "A. B. C. D. E. F. G. H. I. J."
    segs = [{"start": 0, "end": 30, "text": text}]
    result = segment_by_content(segs, min_dur=5.0, max_dur=20.0)
    # Each atomic sentence ~3s. Grouping: 6 sentences (18s), emit, then 4 sentences (12s)
    assert len(result) == 2, f"Expected 2 groups, got {len(result)}"
    for seg in result:
        dur = seg["end"] - seg["start"]
        assert dur <= 20.0 + 1.0, f"Group {seg['text'][:30]}... duration {dur:.1f}s > 20s"


def test_segment_single_long_sentence():
    """One atomic sentence >20s → emitted as-is (no mid-sentence split)."""
    segs = [{"start": 0, "end": 25, "text": "Đây là một câu rất dài không có dấu câu để ngắt vì nó cứ chạy mãi không dừng lại được cho dù nó dài hơn hai mươi giây nhưng vẫn phải giữ nguyên vẹn."}]
    result = segment_by_content(segs, min_dur=5.0, max_dur=20.0)
    assert len(result) == 1
    assert result[0]["end"] - result[0]["start"] > 20.0


def test_segment_tail_short():
    """Tail group <min_dur → still emitted (never dropped)."""
    # 2 sentences, first is 18s, second is 2s (tail < 5s but emitted)
    segs = [{"start": 0, "end": 20, "text": "Câu thứ nhất rất dài ở đây. Câu ngắn."}]
    result = segment_by_content(segs, min_dur=5.0, max_dur=20.0)
    # If 1st sentence alone >= min_dur and 2nd pushes over max_dur → 2 groups
    assert len(result) <= 2
    last = result[-1]
    assert last["text"] == "Câu ngắn." or last["text"].endswith("Câu ngắn.")


def test_segment_music_filtered():
    """Sentences with [âm nhạc] or &gt; are filtered out."""
    segs = [{"start": 0, "end": 10, "text": "[âm nhạc] &gt;&gt; Cổ khí tức cường đại kia gào thét."}]
    result = segment_by_content(segs, min_dur=3.0, max_dur=20.0)
    assert len(result) == 1
    assert "[âm nhạc]" not in result[0]["text"]
    assert "&gt;" not in result[0]["text"]


def test_segment_orphan_first():
    """Only 1 atomic sentence, regardless of duration → emitted as single group."""
    segs = [{"start": 0, "end": 3, "text": "Ngắn."}]
    result = segment_by_content(segs, min_dur=5.0, max_dur=20.0)
    assert len(result) == 1
    assert result[0]["text"] == "Ngắn."


def test_segment_multiple_input_segments():
    """Multiple input segments are flattened into atomic list before grouping."""
    segs = [
        {"start": 0, "end": 4, "text": "Đoạn một. Đoạn hai."},
        {"start": 4, "end": 8, "text": "Đoạn ba. Đoạn bốn."},
    ]
    result = segment_by_content(segs, min_dur=5.0, max_dur=20.0)
    # 4 atomic sentences, each ~2s, total ~8s → all in 1 group
    assert len(result) == 1
    assert "Đoạn một" in result[0]["text"]
    assert "Đoạn bốn" in result[0]["text"]
```

- [ ] **Step 4: Add enhanced dedup test**

```python
def test_dedup_skips_invalid_boundary():
    """Overlap suffix at first position fails boundary check; later position is valid."""
    segs = [
        {"start": 0, "end": 3, "text": "ABC XYZ"},
        {"start": 3, "end": 10, "text": "xABC XYZ and ABC XYZ more"},
    ]
    result = dedup_consecutive_text(segs)
    assert len(result) == 2
    # n=2 "ABC XYZ" first find at pos=1, boundary fail (curr_lower[0]="x")
    # Enhanced: continues searching, finds at pos=12 with valid boundary
    assert result[1]["text"] == "More"
```

- [ ] **Step 5: Add integration test**

```python
def test_pipeline_integration_content_aware():
    """Run the full new pipeline chain on mock data."""
    segments = [
        {"start": 0, "end": 15, "text": "Câu thứ nhất. Câu thứ hai. Câu thứ ba."},
        {"start": 15, "end": 25, "text": "Câu thứ tư dài hơn nhiều so với các câu trước đó."},
    ]
    for seg in segments:
        seg["text"] = clean_text(seg["text"])
    deduped = dedup_consecutive_text(segments)
    fixed = fix_time_overlaps(deduped)
    grouped = segment_by_content(fixed, min_dur=5.0, max_dur=20.0)
    final = [seg for seg in grouped if seg["text"] and (seg["end"] - seg["start"]) >= 2.0 and len(seg["text"]) >= 10]
    assert len(final) >= 1
    for seg in final:
        assert seg["end"] - seg["start"] >= 2.0
        assert len(seg["text"]) >= 10
        assert seg["end"] >= seg["start"]
```

- [ ] **Step 6: Remove obsolete tests and clean up**

Delete these test functions:
- `test_split_by_sentence_short_unchanged`
- `test_split_by_sentence_splits`
- `test_merge_short_merges_into_next`
- `test_merge_short_keeps_normal`

Ensure import list at top of file is clean (no `split_by_sentence`, `merge_short_segments`).

- [ ] **Step 7: Run all tests to verify they pass**

Run: `python -m pytest tests/test_processor.py -v`

Expected: All 18 existing + 9 new = ~23 tests PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_processor.py
git commit -m "test: add segment_by_content tests, remove obsolete split/merge tests

7 new segment_by_content tests covering: basic grouping, overflow,
single long sentence, tail short, music filtered, orphan first,
multiple input segments.
1 enhanced dedup test for multi-position suffix matching.
1 integration test for full pipeline chain.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Update `_analyze2.py` to match new pipeline

**Files:**
- Modify: `_analyze2.py`

- [ ] **Step 1: Update import and pipeline chain**

In `_analyze2.py`, replace:
```python
from tts_pipeline.processor import clean_text, split_by_sentence, dedup_consecutive_text, fix_time_overlaps, merge_short_segments
```

With:
```python
from tts_pipeline.processor import clean_text, dedup_consecutive_text, fix_time_overlaps, segment_by_content
```

Replace the pipeline section (lines 49-63):
```python
for seg in merged:
    seg["text"] = clean_text(seg["text"])

split = split_by_sentence(merged, s.min_segment_dur)
print(f"\nAfter split_by_sentence: {len(split)}", file=log)

deduped = dedup_consecutive_text(split)
...
```

With:
```python
for seg in merged:
    seg["text"] = clean_text(seg["text"])

deduped = dedup_consecutive_text(merged)
print(f"\nAfter dedup: {len(deduped)}", file=log)

fixed = fix_time_overlaps(deduped)
print(f"After fix_time: {len(fixed)}", file=log)

grouped = segment_by_content(fixed, s.min_segment_dur, s.max_segment_dur)
print(f"After segment_by_content: {len(grouped)}", file=log)

fixed2 = fix_time_overlaps(grouped)
print(f"After fix_time (2nd pass): {len(fixed2)}", file=log)

filtered = [seg for seg in fixed2
            if seg["text"] and len(seg["text"]) >= s.min_text_len
            and (seg["end"] - seg["start"]) >= s.min_segment_dur]
print(f"After filter: {len(filtered)}", file=log)
```

Also update the summary line to reference `segment_by_content` instead of `merge_short`.

- [ ] **Step 2: Commit**

```bash
git add _analyze2.py
git commit -m "chore: update _analyze2.py to use segment_by_content pipeline

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Run full pipeline and verify

**Files:**
- Run against the existing downloaded video (`iUGFXuxHNAA`)

- [ ] **Step 1: Run the full pipeline**

```bash
python pipeline_tts.py https://www.youtube.com/watch?v=iUGFXuxHNAA --raw raw --output dataset_test
```

Expected: Pipeline runs to completion with the new pipeline order. No errors.

- [ ] **Step 2: Run the analysis script to verify metrics**

```bash
python _analyze2.py
cat _analysis_detailed.txt
```

Expected metrics should show:
- No segments < 2s
- No segments < 5s where a longer neighbor exists (orphan tail may be < 5s)
- No segment > 20s unless it's a single sentence
- Zero text overlaps between consecutive segments
- No music/noise artifacts ([âm nhạc], &gt;)
- Segment count changed from 446 to a new baseline (expected fewer, better-grouped segments)

- [ ] **Step 3: Check parquet output quality (first 30 samples)**

```python
import pyarrow.parquet as pq
t = pq.read_table("dataset_test/iUGFXuxHNAA_train.parquet")
for i in range(min(30, len(t.column("transcription")))):
    print(f"  {i:3d}: {t.column('transcription')[i].as_py()[:120]}")
```

Manually verify: no short orphan segments, no mid-sentence splits, no overlap between consecutive segments.

- [ ] **Step 4: Check that no overlap exists by running analysis**

```bash
python -c "
import pyarrow.parquet as pq
t = pq.read_table('dataset_test/iUGFXuxHNAA_train.parquet')
txts = t.column('transcription').to_pylist()
overlap_count = 0
for i in range(len(txts)-1):
    a, b = txts[i].split(), txts[i+1].split()
    for n in range(min(len(a), len(b)), 0, -1):
        if ' '.join(a[-n:]).lower() == ' '.join(b[:n]).lower():
            print(f'  OVERLAP-{n}: seg {i} → {i+1}')
            overlap_count += 1
            break
print(f'Total overlaps: {overlap_count}')
"
```

Expected: 0 overlaps

- [ ] **Step 5: Commit any final changes + tag**

```bash
git add pipeline_tts.py  # if any changes needed
git commit -m "chore: update pipeline_tts.py if needed"
git tag -a v2.0 -m "Content-aware segmentation: 5-20s, music filter, dedup enhancement"
```
