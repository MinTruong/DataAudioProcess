"""Tests for VAD interval grouping and text alignment."""
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups


def test_group_empty():
    assert group_vad_intervals([]) == []


def test_group_single():
    iv = [{"start": 0, "end": 3}]
    result = group_vad_intervals(iv, min_dur=5.0, max_dur=20.0)
    # Single short interval stays as-is (tail emit)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 3


def test_group_multiple():
    iv = [
        {"start": 0, "end": 8},
        {"start": 10, "end": 14},
        {"start": 16, "end": 22},
    ]
    result = group_vad_intervals(iv, min_dur=5.0, max_dur=12.0)
    assert len(result) >= 2  # first two (8s+4s=12s max), third is tail


def test_align_text_basic():
    groups = [{"start": 5, "end": 15}]
    cues = [
        {"start": 4, "end": 10, "text": "xin chào"},
        {"start": 10, "end": 16, "text": "các bạn"},
    ]
    result = align_text_to_groups(groups, cues, punctuation=False)
    assert len(result) == 1
    assert "xin chào" in result[0]["text"]
    assert "các bạn" in result[0]["text"]
    assert result[0]["start"] == 5
    assert result[0]["end"] == 15


def test_align_no_overlap():
    groups = [{"start": 100, "end": 110}]
    cues = [{"start": 0, "end": 5, "text": "hello"}]
    result = align_text_to_groups(groups, cues, punctuation=False)
    assert len(result) == 1
    assert result[0]["text"] == ""


def test_align_partial_overlap():
    groups = [{"start": 2, "end": 8}]
    cues = [
        {"start": 0, "end": 4, "text": "alo"},    # 50% overlap
        {"start": 4, "end": 5, "text": "xin"},    # full overlap
        {"start": 9, "end": 12, "text": "chào"},  # no overlap (gap)
    ]
    result = align_text_to_groups(groups, cues, punctuation=False)
    assert "alo" in result[0]["text"]
    assert "xin" in result[0]["text"]
    assert "chào" not in result[0]["text"]
