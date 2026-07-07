#!/usr/bin/env python
"""Debug VAD output - saves results to a file."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from tts_pipeline.config import Settings
from tts_pipeline.downloader import download_video_and_subs
from tts_pipeline.parser import parse_vtt_full
from tts_pipeline.vad import get_speech_intervals
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups

s = Settings()
raw = str(s.raw_dir)

# Parse VTT
raw_cues = parse_vtt_full(os.path.join(raw, "T_zgDuLSIYU.vi.vtt"))
print(f"Raw cues: {len(raw_cues)}")

# VAD
audio_path = os.path.join(raw, "T_zgDuLSIYU.wav")
speech = get_speech_intervals(
    audio_path,
    sample_rate=s.sample_rate,
    threshold=0.3,
    min_speech_dur=0.3,
    min_silence_dur=0.3,
)
print(f"Speech intervals: {len(speech)}")
if speech:
    total_speech = sum(iv["end"] - iv["start"] for iv in speech)
    print(f"Total speech duration: {total_speech:.1f}s")
    print(f"First 3: {speech[:3]}")
else:
    print("WARNING: No speech detected!")
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio duration: {len(audio)/16000:.1f}s")

# Group
groups = group_vad_intervals(speech)
print(f"Groups: {len(groups)}")

# Align
segments = align_text_to_groups(groups, raw_cues, punctuation=True)
print(f"Segments: {len(segments)}")
print(f"With text: {sum(1 for s in segments if s['text'])}")
