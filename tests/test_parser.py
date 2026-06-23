"""Tests for VTT parser."""

import pytest

from tts_pipeline.parser import parse_vtt_full, merge_cues, parse_vtt_time


def test_parse_vtt_time_seconds_only():
    assert parse_vtt_time("00:01.560") == pytest.approx(1.56, abs=0.01)


def test_parse_vtt_time_hms():
    assert parse_vtt_time("00:00:03.350") == pytest.approx(3.35, abs=0.01)


def test_parse_vtt_time_long():
    assert parse_vtt_time("01:02:03.456") == pytest.approx(3723.456, abs=0.01)


def test_merge_cues_empty():
    assert merge_cues([]) == []


def test_merge_cues_single():
    cues = [{"start": 0.0, "end": 5.0, "text": "Hello world."}]
    result = merge_cues(cues)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world."


def test_merge_cues_fragment():
    cues = [
        {"start": 0.0, "end": 3.0, "text": "Hello world."},
        {"start": 3.0, "end": 3.01, "text": "Hello world."},
    ]
    result = merge_cues(cues)
    # fragment (< 0.3s) should extend end
    assert result[0]["end"] >= 3.01
