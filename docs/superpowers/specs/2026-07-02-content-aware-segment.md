# Content-Aware Sentence Segmentation for Vietnamese TTS Pipeline

**Date:** 2026-07-02
**Status:** Draft
**Author:** Brainstorming session

## 🔴 Problem

The pipeline currently produces ~446 segments, but quality issues remain:

1. **Over-splitting:** `split_by_sentence` cuts at every `. ! ?`, producing short sentence fragments that are incomplete in meaning (e.g., `"Rồi vậy thì cũng không vài vòng nữa."` as a standalone segment).
2. **Segments too short:** Many segments are 2-4s despite `merge_short_segments` trying to catch them, because the merge only fires for `<2s` or `<10 chars`, leaving orphans.
3. **Duration not considered for grouping:** Split is purely punctuation-driven — duration is only used for time proration, not as a grouping criterion.
4. **Remaining artifacts:** Music tags (`[âm nhạc]`), residual text overlaps, truncated segments.

**Root cause:** The pipeline splits text into sentences first (punctuation-only), then tries to merge back short ones — but the merge heuristics are too weak to recover from an over-aggressive split.

## 🎯 Goal

Replace `split_by_sentence` + `merge_short_segments` with a single content-aware grouping function that:

- Produces **segments of 5–20 seconds** duration
- Each segment is **a complete semantic unit** (one or more full sentences)
- No sentence is left as a short orphan (<5s) unless it is physically the only segment
- Handles time proration correctly even after grouping
- Eliminates music/noise artifacts before grouping

## 🧱 Design

### Pipeline Order (Updated)

```
merge → clean → dedup → fix_time_overlaps → segment_by_content → 
fix_time_overlaps → filter → export
```

Key change: `split_by_sentence` and `merge_short_segments` **removed**. `dedup` + `fix_time` moved **before** segmentation. An extra `fix_time_overlaps` pass added after grouping to ensure clean time boundaries.

### Module: `segment_by_content`

Implemented in `tts_pipeline/processor.py` as `segment_by_content()`.

```
segment_by_content(segments: list[dict], 
                   min_dur: float = 5.0, 
                   max_dur: float = 20.0,
                   min_chars: int = 10) -> list[dict]
```

#### Phase 1 — Atomic Sentence Extraction

For each input segment:
1. Split text into atomic sentences via regex `(?<=[.!?])\s+`
2. Prorate time per sentence using character ratio: `char_ratio = dur / total_chars`
3. Each sentence becomes `{start, end, text}`, flat list
4. **Music/noise filter:** atomic sentences containing `[âm nhạc]`, `[music]`, `&gt;` → skipped (not added to list)
5. Empty sentences after filtering → skipped

#### Phase 2 — Greedy Duration-Based Grouping

Traverse the atomic sentence list left-to-right:

```
group = [first_sentence]
for each remaining sentence:
    group_dur = end - start of group
    add_dur = duration of candidate sentence
    
    if group_dur < min_dur:
        # Must append — no orphan allowed
        group.append(candidate)
    elif group_dur + add_dur > max_dur:
        # Emit current group, start new
        emit(group)
        group = [candidate]
    else:
        # Room to grow
        group.append(candidate)

# Handle tail group
emit(group)  # always — tail emits even if < min_dur
```

#### Phase 3 — Time Calculation for Groups

- `start` = start of first atomic sentence in group
- `end` = end of last atomic sentence in group
- No interpolation, just boundary extension

### Edge Cases

| Case | Handling |
|------|----------|
| One atomic sentence >20s | Emit as-is (do not split mid-sentence) |
| Tail group <5s with multiple prior groups | Emit as-is (it's the end of video) |
| Single group spans entire segment | Emit as-is (no earlier group to merge into, no later group to append) |
| Music/noise artifacts | Filtered in Phase 1 |
| Empty segment after all filters | Filtered out in the final `filter` stage |
| Time discontinuity after grouping | Second `fix_time_overlaps` pass heals it |

### Dedup Enhancement (Bonus Fix)

Current `dedup_consecutive_text` searches for suffix of prev text anywhere in current text, stripping through it. It uses a word-boundary guard (`not curr_lower[pos-1].isspace()`) that causes false negatives: e.g., `"đi. La Lạc Ma quân..."` in seg 27 reappears mid-sentence in seg 28 as `"đi. La Lạc Ma quân..."` but preceded by non-whitespace? No, it IS preceded by whitespace in the actual case — but the `pos` found is > 0 so it checks boundary. The real issue: the current code only checks the FIRST match position. If the suffix appears once near the start (at position where the preceding char is not a space) and once later (at a valid position), it misses the later valid one.

Enhancement: **Iterate all match positions** for each suffix length, not just the first. Pick the one with the smallest `pos + len(suffix)` (i.e., strip the least amount of current text) to avoid stripping too aggressively.

### Testing Plan

| Test | Description |
|------|-------------|
| `test_segment_grouping_basic` | 3 atomic sentences of 2s each → 1 group of 6s |
| `test_segment_grouping_overflow` | 2 groups where 2nd sentence exceeds max_dur |
| `test_segment_single_long_sentence` | One atomic sentence >20s emitted as-is |
| `test_segment_tail_short` | Tail group <5s emitted |
| `test_segment_music_filtered` | Sentence with `[âm nhạc]` removed |
| `test_segment_orphan_first` | Only 1 atomic sentence, <5s → emitted |
| `test_dedup_enhanced` | Overlap with leading word before match position |
| `test_pipeline_integration` | Full pipeline: merge → clean → dedup → fix_time → segment → fix_time → filter |

## 📊 Success Criteria

- **No segment < 2s** (hard lower bound via filter)
- **No segment < 5s** where a longer neighbor exists to merge into (soft: orphan tail may be <5s)
- **No segment > 20s** unless it's a single sentence
- **No music/noise artifacts** in output
- **Zero text overlaps** between consecutive segments
- **Test count:** All existing tests pass + 7 new tests + 1 integration test

## 📋 Implementation Order

1. Add `segment_by_content()` to `processor.py`
2. Update `cli.py` pipeline order (remove `split_by_sentence`, `merge_short_segments`; add `segment_by_content`)
3. Enhance `dedup_consecutive_text()` — relax word-boundary constraint
4. Add music/noise filter to `clean_text()` or inline in `segment_by_content`
5. Add 8 new tests to `tests/test_processor.py`
6. Update `tests/test_processor.py` to remove split_by_sentence/merge_short_segments tests
7. Run full pipeline and verify metrics
8. Commit
