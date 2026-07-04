"""Show how segments are produced from VTT."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.config import Settings
from tts_pipeline.processor import clean_text, split_by_sentence, dedup_consecutive_text, fix_time_overlaps, merge_short_segments

s = Settings()
raw = parse_vtt_full("raw/iUGFXuxHNAA.vi.vtt")
merged = merge_cues(raw, s)

print(f"RAW VTT cues: {len(raw)}")
print(f"After merge:  {len(merged)}")
print()

print("=== 5 cues đầu sau merge ===")
for i, seg in enumerate(merged[:5]):
    dur = seg["end"] - seg["start"]
    print(f"  {i}: [{seg['start']:.1f} - {seg['end']:.1f}] ({dur:.1f}s) {seg['text'][:100]}")

# After clean + split
for seg in merged:
    seg["text"] = clean_text(seg["text"])

split = split_by_sentence(merged, s.min_segment_dur)
print(f"\nAfter split_by_sentence: {len(split)}")
print("=== 10 segments đầu sau split ===")
for i, seg in enumerate(split[:10]):
    dur = seg["end"] - seg["start"]
    print(f"  {i}: [{seg['start']:.1f} - {seg['end']:.1f}] ({dur:.1f}s len={len(seg['text']):3d}) {seg['text'][:100]}")

# After dedup
deduped = dedup_consecutive_text(split)
print(f"\nAfter dedup: {len(deduped)}")
overlap_count = sum(1 for i in range(len(deduped)-1) if deduped[i]['text'].split()[-3:] == deduped[i+1]['text'].split()[:3] and len(deduped[i+1]['text'].split()) >=3)
print(f"  Still overlapping (3+ word): {overlap_count}")
print("=== 10 segments đầu sau dedup ===")
for i, seg in enumerate(deduped[:10]):
    dur = seg["end"] - seg["start"]
    print(f"  {i}: [{seg['start']:.1f} - {seg['end']:.1f}] ({dur:.1f}s len={len(seg['text']):3d}) {seg['text'][:100]}")

# After fix_time + merge_short
fixed = fix_time_overlaps(deduped)
merged_short = merge_short_segments(fixed, s.min_segment_dur, s.min_text_len)
filtered = [seg for seg in merged_short if seg['text'] and len(seg['text']) >= s.min_text_len and (seg['end'] - seg['start']) >= s.min_segment_dur]
print(f"\nAfter merge_short: {len(merged_short)}")
print(f"After filter: {len(filtered)}")
print("=== 10 segments đầu sau filter ===")
for i, seg in enumerate(filtered[:10]):
    dur = seg["end"] - seg["start"]
    print(f"  {i}: [{seg['start']:.1f} - {seg['end']:.1f}] ({dur:.1f}s len={len(seg['text']):3d}) {seg['text'][:100]}")
