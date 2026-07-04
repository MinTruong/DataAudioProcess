"""Debug new merge logic."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.config import Settings

s = Settings()
raw = parse_vtt_full("raw/iUGFXuxHNAA.vi.vtt")
print(f"RAW cues: {len(raw)}")
print()

# Show first 15 raw cues
print("=== 15 cues đầu (raw, stripped) ===")
for i, cue in enumerate(raw[:15]):
    dur = cue["end"] - cue["start"]
    print(f"  {i:3d}: [{cue['start']:7.3f} - {cue['end']:7.3f}] ({dur:5.3f}s) {cue['text'][:80]}")

merged = merge_cues(raw, s)
print(f"\nAfter merge: {len(merged)}")
print()
print("=== 10 segments đầu sau merge mới ===")
for i, seg in enumerate(merged[:10]):
    dur = seg["end"] - seg["start"]
    print(f"  {i}: [{seg['start']:.1f} - {seg['end']:.1f}] ({dur:.1f}s) {seg['text'][:100]}")

# Compare with old: 1350 segments, with bad overlaps
# Check for any obvious overlap
overlaps = 0
for i in range(len(merged)-1):
    a_words = merged[i]["text"].split()
    b_words = merged[i+1]["text"].split()
    for n in range(min(len(a_words), len(b_words)), 0, -1):
        if " ".join(a_words[-n:]).lower() == " ".join(b_words[:n]).lower():
            overlaps += 1
            break
print(f"\nOverlap count (any suffix/prefix match): {overlaps}")
