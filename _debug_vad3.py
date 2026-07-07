#!/usr/bin/env python
"""Deep debug: simulate what exporter does."""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from tts_pipeline.config import Settings
from tts_pipeline.parser import parse_vtt_full
from tts_pipeline.vad import get_speech_intervals
from tts_pipeline.vad_aligner import group_vad_intervals, align_text_to_groups
from tts_pipeline.processor import clean_text
from tts_pipeline.punctuator import restore_punctuation

s = Settings()
raw = "raw"

raw_cues = parse_vtt_full(os.path.join(raw, "T_zgDuLSIYU.vi.vtt"))
speech = get_speech_intervals(os.path.join(raw, "T_zgDuLSIYU.wav"), sample_rate=22050, threshold=0.3)
groups = group_vad_intervals(speech, min_dur=5, max_dur=20)
segments = align_text_to_groups(groups, raw_cues, punctuation=True)

# Debug: check each segment against exporter filters
fail_dur = 0
fail_text = 0
fail_clean_text = 0
ok = 0

for idx, seg in enumerate(segments):
    dur = seg["end"] - seg["start"]
    text = seg.get("text", "") or ""

    # Simulate exporter checks
    if dur < s.min_segment_dur or dur > s.max_segment_dur:
        fail_dur += 1
        if idx < 5 or dur > 20:
            print(f"[{idx}] FAIL dur={dur:.1f}s (range {s.min_segment_dur}-{s.max_segment_dur})")
        continue

    if not text or len(text) < s.min_text_len or len(text) > s.max_text_len:
        fail_text += 1
        if idx < 5 or len(text) > s.max_text_len:
            print(f"[{idx}] FAIL text_len={len(text)} (range {s.min_text_len}-{s.max_text_len}) text={text[:80]}...")
        continue

    clean = clean_text(text)
    if not clean or len(clean) < s.min_text_len:
        fail_clean_text += 1
        if idx < 5:
            print(f"[{idx}] FAIL clean_text_len={len(clean) if clean else 0}")
        continue

    ok += 1

print(f"\nTotal: {len(segments)}, OK: {ok}, Fail dur: {fail_dur}, Fail text: {fail_text}, Fail clean: {fail_clean_text}")
print(f"s.min_text_len={s.min_text_len}, s.max_text_len={s.max_text_len}")
