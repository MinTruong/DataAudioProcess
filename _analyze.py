"""Analyze segmentation issues in detail."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyarrow.parquet as pq
from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.config import Settings
from tts_pipeline.processor import clean_text, split_by_sentence, dedup_consecutive_text, fix_time_overlaps, merge_short_segments

s = Settings()

# === STAGE 1: RAW VTT STRUCTURE ===
print("=" * 70)
print("STAGE 1: RAW VTT CUE PATTERN (first 20 cues)")
print("=" * 70)
raw = parse_vtt_full("raw/iUGFXuxHNAA.vi.vtt")
for i, cue in enumerate(raw[:20]):
    dur = cue["end"] - cue["start"]
    is_frag = dur < s.fragment_threshold
    marker = " [FRAG]" if is_frag else "       "
    print(f"  {i:3d}{marker} {cue['start']:7.3f} - {cue['end']:7.3f} ({dur:5.3f}s) {cue['text'][:80]}")

# === STAGE 2: MERGE QUALITY ===
print()
print("=" * 70)
print("STAGE 2: MERGE OUTPUT (all segments)")
print("=" * 70)
merged = merge_cues(raw, s)
print(f"Total after merge: {len(merged)}")
for i, seg in enumerate(merged):
    dur = seg["end"] - seg["start"]
    print(f"  {i:3d}: [{seg['start']:7.2f} - {seg['end']:7.2f}] ({dur:5.1f}s) {seg['text'][:120]}")

# === STAGE 3: PIPELINE CHAIN RESULTS ===
print()
print("=" * 70)
print("STAGE 3: FULL PIPELINE CHAIN")
print("=" * 70)

for seg in merged:
    seg["text"] = clean_text(seg["text"])

print(f"After clean: {len(merged)}")

split = split_by_sentence(merged, s.min_segment_dur)
print(f"After split_by_sentence: {len(split)}")

deduped = dedup_consecutive_text(split)
print(f"After dedup: {len(deduped)} (removed {len(split)-len(deduped)})")

fixed = fix_time_overlaps(deduped)
print(f"After fix_time_overlaps: {len(fixed)}")

short_merged = merge_short_segments(fixed, s.min_segment_dur, s.min_text_len)
print(f"After merge_short: {len(short_merged)}")

filtered = [seg for seg in short_merged
            if seg["text"]
            and len(seg["text"]) >= s.min_text_len
            and (seg["end"] - seg["start"]) >= s.min_segment_dur]
print(f"After filter: {len(filtered)}")
print()

# Show all segments with issues highlighted
print("=" * 70)
print("FINAL SEGMENTS (ISSUES MARKED)")
print("=" * 70)

# Check for overlaps
for i in range(len(filtered)):
    dur = filtered[i]["end"] - filtered[i]["start"]
    text = filtered[i]["text"]
    issues = []

    if dur < s.min_segment_dur:
        issues.append(f"DUR<{s.min_segment_dur}")
    if len(text) < s.min_text_len:
        issues.append(f"LEN<{s.min_text_len}")

    # Check text overlap with next segment
    if i < len(filtered) - 1:
        a_words = text.split()
        b_words = filtered[i+1]["text"].split()
        for n in range(min(len(a_words), len(b_words)), 0, -1):
            if " ".join(a_words[-n:]).lower() == " ".join(b_words[:n]).lower():
                issues.append(f"OVERLAP-{n}")
                break

    # Check if segment ends mid-sentence (no punctuation)
    if text and text[-1] not in ".!?" and text[-1] != '"':
        issues.append("TRUNCATED")

    marker = " *** " if issues else "      "
    issue_str = f"[{', '.join(issues)}]" if issues else ""
    print(f"  {i:3d}{marker} {dur:5.1f}s len={len(text):3d} {issue_str} {text[:120]}")

print()
# Count issues
total = len(filtered)
short_count = sum(1 for s in filtered if s["end"]-s["start"] < s.min_segment_dur)
short_text = sum(1 for s in filtered if len(s["text"]) < s.min_text_len)
overlap = 0
truncated = 0
for i in range(len(filtered)):
    if i < len(filtered) - 1:
        a = filtered[i]["text"].split()
        b = filtered[i+1]["text"].split()
        for n in range(min(len(a), len(b)), 0, -1):
            if " ".join(a[-n:]).lower() == " ".join(b[:n]).lower():
                overlap += 1
                break
    if filtered[i]["text"][-1] not in ".!?\"":
        truncated += 1

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total segments:   {total}")
print(f"  Short duration:   {short_count}")
print(f"  Short text:       {short_text}")
print(f"  Text overlaps:    {overlap}")
print(f"  Truncated (no .): {truncated}")
