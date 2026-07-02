"""Tests for text processor."""

from tts_pipeline.processor import (
    clean_text,
    dedup_consecutive_text,
    fix_time_overlaps,
    segment_by_content,
)


def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_strips_edges():
    assert clean_text("\n\nfoo\n\n") == "foo"


def test_dedup_consecutive_no_change():
    segs = [{"start": 0, "end": 2, "text": "Hello world."}]
    result = dedup_consecutive_text(segs)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world."


def test_dedup_removes_three_word_overlap():
    segs = [
        {"start": 0, "end": 5, "text": "Một hai ba bốn năm"},
        {"start": 5, "end": 10, "text": "bốn năm sáu bảy tám"},
    ]
    result = dedup_consecutive_text(segs)
    assert len(result) == 2
    # First letter is capitalized by dedup logic
    assert result[1]["text"] == "Sáu bảy tám"


def test_dedup_removes_one_word_overlap():
    segs = [
        {"start": 0, "end": 3, "text": "Hello world foo"},
        {"start": 3, "end": 6, "text": "world foo bar baz"},
    ]
    result = dedup_consecutive_text(segs)
    assert len(result) == 2
    assert result[1]["text"] == "Bar baz"


def test_dedup_removes_long_overlap():
    """Bắt overlap dài >3 từ mà logic cũ bỏ sót."""
    segs = [
        {"start": 0, "end": 5, "text": "với kênh Bách Audio Vấn Đậu Tông."},
        {"start": 5, "end": 10,
         "text": "lại với kênh Bách Audio Vấn Đậu Tông. Thì tiếp tục ngày hôm nay."},
    ]
    result = dedup_consecutive_text(segs)
    assert len(result) == 2
    assert "lại với kênh Bách Audio Vấn Đậu Tông" not in result[0]["text"]
    assert result[1]["text"].startswith("Thì")


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


