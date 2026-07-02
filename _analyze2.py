"""Analyze segmentation issues - save to file to avoid output pollution."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tts_pipeline.parser import parse_vtt_full, merge_cues
from tts_pipeline.config import Settings
from tts_pipeline.processor import clean_text, dedup_consecutive_text, fix_time_overlaps, segment_by_content

s = Settings()

# Redirect prints to file
log = open("D:\\MinhTh_code\\DataAudioProcess\\_analysis_detailed.txt", "w", encoding="utf-8")

def p(*a, **kw):
    print(*a, **kw)
    print(*a, file=log, **kw)

raw = parse_vtt_full("raw/iUGFXuxHNAA.vi.vtt")

# === STAGE 1: COUNT RAW CUE TYPES ===
word_cues = [c for c in raw if c["end"] - c["start"] >= s.fragment_threshold]
frag_cues = [c for c in raw if c["end"] - c["start"] < s.fragment_threshold]
total = len(raw)
p(f"Raw VTT: {total} cues total")
p(f"  Word-level (>={s.fragment_threshold}s): {len(word_cues)}")
p(f"  Fragments (<{s.fragment_threshold}s): {len(frag_cues)}")
p(f"  Ratio fragments/total: {len(frag_cues)/total*100:.0f}%")

# === STAGE 2: ANALYZE MERGE ===
merged = merge_cues(raw, s)
p(f"\nAfter merge: {len(merged)} segments")

# Merge segment stats
durations = [m["end"] - m["start"] for m in merged]
text_lens = [len(m["text"]) for m in merged]
p(f"  Duration: min={min(durations):.1f}s max={max(durations):.1f}s mean={sum(durations)/len(durations):.1f}s")
p(f"  Text len: min={min(text_lens)} max={max(text_lens)} mean={sum(text_lens)/len(text_lens):.0f}")

# Check merge truncation (segments ending without punctuation)
trunc_merge = [m for m in merged if m["text"] and m["text"][-1] not in ".!?\"" and not m["text"][-1].isupper()]
p(f"  Truncated (no period, end lowercase): {len(trunc_merge)}")
for m in trunc_merge[:10]:
    p(f"    [{m['start']:.1f}-{m['end']:.1f}] ...{m['text'][-40:]}")

# === STAGE 3: FULL PIPELINE ===
for seg in merged:
    seg["text"] = clean_text(seg["text"])

deduped = dedup_consecutive_text(merged)
p(f"\nAfter dedup: {len(deduped)} segments")

fixed = fix_time_overlaps(deduped)
p(f"After fix_time: {len(fixed)} segments")

grouped = segment_by_content(fixed, s.min_segment_dur, s.max_segment_dur)
p(f"After segment_by_content: {len(grouped)} segments")

fixed2 = fix_time_overlaps(grouped)
p(f"After fix_time (2nd pass): {len(fixed2)} segments")

filtered = [seg for seg in fixed2
            if seg["text"] and len(seg["text"]) >= s.min_text_len
            and (seg["end"] - seg["start"]) >= s.min_segment_dur]
p(f"After filter: {len(filtered)} segments")

# === STAGE 4: DETAILED ISSUES IN FINAL OUTPUT ===
p(f"\n{'='*70}")
p("DETAILED ISSUE ANALYSIS")
p(f"{'='*70}")

# 1. Text overlaps
overlap_list = []
for i in range(len(filtered)-1):
    a = filtered[i]["text"].split()
    b = filtered[i+1]["text"].split()
    for n in range(min(len(a), len(b)), 0, -1):
        if " ".join(a[-n:]).lower() == " ".join(b[:n]).lower():
            overlap_list.append((i, n, filtered[i]["text"][-40:], filtered[i+1]["text"][:80]))
            break

p(f"\n1. TEXT OVERLAPS: {len(overlap_list)} pairs")
for idx, n, tail, head in overlap_list[:15]:
    p(f"  #{idx}->#{idx+1} (overlap {n} từ):")
    p(f"    prev[-40]: ...{tail}")
    p(f"    curr[:80]: {head}")
if len(overlap_list) > 15:
    p(f"  ... and {len(overlap_list)-15} more")

# 2. Truncated sentences
trunc_list = [s for s in filtered if s["text"][-1] not in ".!?\""]
p(f"\n2. TRUNCATED (no sentence ending): {len(trunc_list)} segments")
for s in trunc_list[:15]:
    p(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text'][:120]}")

# 3. Short segments
short_dur = [s for s in filtered if s["end"]-s["start"] < 2.0]
short_txt = [s for s in filtered if len(s["text"]) < 10]
p(f"\n3. SHORT SEGMENTS")
p(f"  Duration < 2.0s: {len(short_dur)}")
p(f"  Text < 10 chars: {len(short_txt)}")

# 4. Very long segments
long_seg = [s for s in filtered if len(s["text"]) > 200]
p(f"\n4. VERY LONG TEXT (>200 chars): {len(long_seg)}")
for s in long_seg[:10]:
    p(f"  [{s['start']:.1f}-{s['end']:.1f}] len={len(s['text'])} {s['text'][:150]}...")

# 5. Music/noise artifacts
music = [s for s in filtered if "[âm nhạc]" in s["text"].lower() or "&gt;" in s["text"]]
p(f"\n5. MUSIC/NOISE ARTIFACTS: {len(music)}")
for s in music[:5]:
    p(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text'][:100]}")

# 6. Duplicate/overlap in parquet (BOTH sides)
p(f"\n6. PARQUET SAMPLE (first 60 train segments)")
import pyarrow.parquet as pq
t = pq.read_table("dataset/iUGFXuxHNAA_train.parquet")
train_txt = t.column("transcription").to_pylist()
for i in range(min(60, len(train_txt))):
    txt = train_txt[i]
    issues = []
    if len(txt) < 10:
        issues.append("SHORT")
    if txt[-1] not in ".!?\"":
        issues.append("TRUNC")
    if i < len(train_txt) - 1:
        a = txt.split()
        b = train_txt[i+1].split()
        for n in range(min(len(a), len(b)), 0, -1):
            if " ".join(a[-n:]).lower() == " ".join(b[:n]).lower():
                issues.append(f"OVLP-{n}")
                break
    tag = " ***" if issues else ""
    p(f"  {i:3d}{tag} {txt[:150]}")

p(f"\n{'='*70}")
p("SUMMARY STATS")
p(f"{'='*70}")
p(f"  Total segments:       {len(filtered)}")
p(f"  Text overlaps:        {len(overlap_list)}")
p(f"  Truncated (no .!?):   {len(trunc_list)}")
p(f"  Short duration:       {len(short_dur)}")
p(f"  Short text:           {len(short_txt)}")
p(f"  Long text (>200):     {len(long_seg)}")
p(f"  Music/noise:          {len(music)}")
p(f"  Avg text length:      {sum(len(s['text']) for s in filtered)/len(filtered):.0f} chars")
p(f"  Avg duration:         {sum(s['end']-s['start'] for s in filtered)/len(filtered):.1f}s")

log.close()
