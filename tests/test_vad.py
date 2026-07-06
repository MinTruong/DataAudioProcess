"""Tests for VAD speech interval detection."""
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch

from tts_pipeline.vad import get_speech_intervals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_speech_fixture_path():
    """Return path to the real speech test file bundled with Silero-VAD."""
    return os.path.join(
        torch.hub.get_dir(),
        "snakers4_silero-vad_master",
        "tests",
        "data",
        "test.wav",
    )


def _create_speech_wav(duration: float, sample_rate: int = 22050) -> str:
    """Copy a snippet of the bundled real speech WAV to a temp file."""
    src = _get_speech_fixture_path()
    audio, sr = librosa.load(src, sr=sample_rate, mono=True)
    n = int(duration * sample_rate)
    snippet = audio[:n]
    fd, path = tempfile.mkstemp(suffix="_test_vad.wav")
    os.close(fd)
    sf.write(path, snippet, sample_rate)
    return path


def _create_speech_with_silence_gap(
    total_dur: float, gap_start: float, gap_end: float, sample_rate: int = 22050
) -> str:
    """Take a real speech snippet and zero out a region to simulate silence."""
    src = _get_speech_fixture_path()
    audio, sr = librosa.load(src, sr=sample_rate, mono=True)
    n = int(total_dur * sample_rate)
    snippet = audio[:n].copy()
    # Zero out the requested gap
    s = int(gap_start * sample_rate)
    e = int(gap_end * sample_rate)
    snippet[s:e] = 0.0
    fd, path = tempfile.mkstemp(suffix="_test_vad_gap.wav")
    os.close(fd)
    sf.write(path, snippet, sample_rate)
    return path


def _create_silent_wav(duration: float, sample_rate: int = 22050) -> str:
    """Create a silent WAV file."""
    audio = np.zeros(int(sample_rate * duration))
    fd, path = tempfile.mkstemp(suffix="_test_vad_silence.wav")
    os.close(fd)
    sf.write(path, audio, sample_rate)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vad_returns_intervals():
    """get_speech_intervals returns a list of {start, end} dicts."""
    path = _create_speech_wav(3.0)
    try:
        intervals = get_speech_intervals(path)
        assert isinstance(intervals, list)
        assert len(intervals) > 0
        for iv in intervals:
            assert "start" in iv and "end" in iv
            assert iv["end"] > iv["start"]
    finally:
        os.unlink(path)


def test_vad_detects_silence_gap():
    """Intervals should be split at the silence gap."""
    # Use a 5-second real-speech snippet with 1 s silence in the middle
    path = _create_speech_with_silence_gap(5.0, 2.0, 3.0)
    try:
        intervals = get_speech_intervals(path, min_silence_dur=0.5)
        # Should get >= 2 intervals thanks to the silence gap
        assert len(intervals) >= 2, f"Expected >=2 intervals, got {len(intervals)}"
    finally:
        os.unlink(path)


def test_vad_returns_empty_for_silence():
    """All-silence audio returns empty list."""
    path = _create_silent_wav(2.0)
    try:
        intervals = get_speech_intervals(path)
        assert intervals == []
    finally:
        os.unlink(path)
