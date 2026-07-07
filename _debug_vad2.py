#!/usr/bin/env python
"""Debug: why only 21 segments? Check overlap between VAD groups and VTT cues."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(__file__))

from tts_pipeline.config import Settings
from tts_pipeline.parser import parse_vtt_full
from tts_pipeline.vad import get_speech_intervals
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups, _overlap_ratio

s = Settings()
raw = str(s.raw_dir)

raw_cues = parse_vtt_full(os.path.join(raw, "T_zgDuLSIYU.vi.vtt"))
print(f"Raw VTT cues: {len(raw_cues)}")

speech = get_speech_intervals(
    os.path.join(raw, "T_zgDuLSIYU.wav"),
    sample_rate=s.sample_rate,
    threshold=0.3,
    min_speech_dur=0.3,
    min_silence_dur=0.3,
)
print(f"Speech intervals: {len(speech)}")

groups = group_vad_intervals(speech, min_dur=5.0, max_dur=20.0)
print(f"Groups: {len(groups)}")

# Check alignment for first 10 groups
for i, g in enumerate(groups[:10]):
    overlap_cues = []
    for cue in raw_cues:
        ratio = _overlap_ratio(cue["start"], cue["end"], g["start"], g["end"])
        if ratio > 0.1:
            overlap_cues.append((cue["text"][:40], ratio))
    g_dur = g["end"] - g["start"]
    print(f"Group[{i:03d}] t={g['start']:.1f}-{g['end']:.1f}s (dur={g_dur:.1f}s) cues={len(overlap_cues)}")
    if overlap_cues:
        for text, ratio in overlap_cues[:3]:
            print(f"  cue overlap={ratio:.2f}: {text}")
    else:
        print(f"  NO CUES OVERLAP!")
