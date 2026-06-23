"""Tests for text processor."""

from tts_pipeline.processor import clean_text, dedup_consecutive_text, split_long_segments


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


def test_split_long_segments_short_unchanged():
    segs = [{"start": 0, "end": 5, "text": "Short text."}]
    result = split_long_segments(segs, max_dur=20.0)
    assert len(result) == 1


def test_split_long_segments_splits():
    segs = [{"start": 0, "end": 25, "text": "Câu thứ nhất. Câu thứ hai. Câu thứ ba."}]
    result = split_long_segments(segs, max_dur=20.0)
    assert len(result) >= 2
