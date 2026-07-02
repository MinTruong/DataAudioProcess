"""Text cleaning, deduplication, and segment splitting."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


_MUSIC_PATTERNS = ["[âm nhạc]", "[nhạc]", "[music]", "&gt;"]

# Normalized patterns for matching (spaces removed to match .replace(" ", "") on input)
_NORMALIZED_MUSIC_PATTERNS = [p.replace(" ", "") for p in _MUSIC_PATTERNS]


def _has_music_noise(text: str) -> bool:
    """Check if text contains music/noise/copyright artifacts to filter out."""
    lower = text.lower().replace(" ", "")
    for p in _NORMALIZED_MUSIC_PATTERNS:
        if p in lower:
            return True
    return False


def segment_by_content(
    segments: list[dict],
    min_dur: float = 5.0,
    max_dur: float = 20.0,
) -> list[dict]:
    """Group atomic sentences into 5-20s segments with complete semantic units.

    Phase 1 -- Split each segment into atomic sentences by punctuation, prorate
    time by character ratio. Filter out music/noise artifacts.

    Phase 2 -- Greedy left-to-right grouping:
    - If group duration < min_dur: append candidate unconditionally
    - If adding candidate would exceed max_dur: emit current group, start new
    - Otherwise: append candidate
    Tail group is always emitted (no backward merging into previous group).
    """
    # Phase 1: Extract atomic sentences with prorated time
    atoms: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg["text"]
        if not text.strip():
            continue

        parts = re.split(r"(?<=[.!?])\s+", text)
        total_chars = sum(len(p) for p in parts)
        if total_chars == 0:
            continue

        char_ratio = dur / total_chars
        cur_start = seg["start"]

        for part in parts:
            part = part.strip()
            if not part or _has_music_noise(part):
                continue
            part_dur = len(part) * char_ratio
            atoms.append({
                "start": cur_start,
                "end": cur_start + part_dur,
                "text": part,
            })
            cur_start += part_dur

    # Phase 2: Greedy duration-based grouping
    if not atoms:
        return []

    groups: list[list[dict]] = []
    current = [atoms[0]]

    for atom in atoms[1:]:
        group_dur = current[-1]["end"] - current[0]["start"]
        add_dur = atom["end"] - atom["start"]

        if group_dur < min_dur:
            current.append(atom)          # must append -- no orphan
        elif group_dur + add_dur > max_dur:
            groups.append(current)         # emit, start fresh
            current = [atom]
        else:
            current.append(atom)           # room to grow

    groups.append(current)                 # always emit tail

    output = []
    for group in groups:
        joined = " ".join(a["text"] for a in group)
        cleaned = clean_text(joined)
        output.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": cleaned,
        })

    return output


def dedup_consecutive_text(segments: list[dict]) -> list[dict]:
    """Remove overlapping text between consecutive segments.

    YouTube captions cause the last words of segment N to repeat
    at the start of segment N+1 (sometimes with extra leading words
    like "lại", "và", "rồi"). Searches for the longest suffix of prev
    within the first part of current, then strips everything through it.
    """
    result: list[dict] = []
    for seg in segments:
        text = seg["text"]

        if result:
            prev_text = result[-1]["text"]
            prev_words = prev_text.split()
            curr_lower = text.lower()

            # Find longest suffix of prev found near the start of current
            limit = min(len(prev_words), 15)
            for n in range(limit, 0, -1):
                prev_suffix = " ".join(prev_words[-n:])
                pos = curr_lower.find(prev_suffix.lower())
                if pos == -1:
                    continue
                # Must be at word boundary
                if pos > 0 and not curr_lower[pos - 1].isspace():
                    continue
                # Strip from start of current through the end of the overlap
                text = text[pos + len(prev_suffix):].strip()
                text = text.lstrip(",").strip()
                if not text:
                    continue
                text = text[0].upper() + text[1:]
                break

        result.append({**seg, "text": text})

    return result


def split_by_sentence(
    segments: list[dict], min_dur: float = 2.0
) -> list[dict]:
    """Split all segments into individual sentences.

    Each segment output = one sentence, with time prorated by character count.
    Skips splitting if text has no sentence boundary (keeps original segment).
    """
    result: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg["text"]

        # Try sentence split (. ! ?), fallback to comma split
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) < 2:
            parts = re.split(r"(?<=[,])\s+", text)
        if len(parts) < 2:
            result.append(seg)
            continue

        total_chars = sum(len(s) for s in parts)
        if total_chars == 0:
            result.append(seg)
            continue

        char_ratio = dur / total_chars
        current_start = seg["start"]

        for sent in parts:
            sent = sent.strip()
            if not sent:
                continue
            sent_dur = max(len(sent) * char_ratio, min_dur)
            result.append({
                "start": current_start,
                "end": current_start + sent_dur,
                "text": sent,
            })
            current_start += sent_dur

    return result


def fix_time_overlaps(segments: list[dict]) -> list[dict]:
    """Fix overlapping time between consecutive segments.

    If segment N+1 has start < end of segment N, push start forward.
    """
    result: list[dict] = []
    for seg in segments:
        if result:
            prev_end = result[-1]["end"]
            if seg["start"] < prev_end:
                seg = {**seg, "start": prev_end}
            if seg["end"] - seg["start"] < 0.1:
                continue
        result.append(seg)
    return result


def merge_short_segments(
    segments: list[dict],
    min_dur: float = 2.0,
    min_text_len: int = 10,
) -> list[dict]:
    """Merge short sentences into the following segment.

    A segment is 'short' if duration < min_dur or text length < min_text_len.
    Short segments are merged forward: text joined, time extended.
    Accumulates consecutive short segments, then merges the whole block
    into the first non-short segment that follows.
    The last segment is kept as-is even if short.
    """
    if not segments:
        return []

    result: list[dict] = []
    buffer = None  # accumulated short segment(s)

    for seg in segments:
        dur = seg["end"] - seg["start"]
        text_len = len(seg["text"])
        is_short = dur < min_dur or text_len < min_text_len

        if is_short and buffer is None:
            buffer = seg
        elif is_short and buffer is not None:
            gap = seg["start"] - buffer["end"]
            spacer = " " if gap < 2.0 else ". "
            buffer = {
                "start": buffer["start"],
                "end": seg["end"],
                "text": buffer["text"].rstrip() + spacer + seg["text"],
            }
        elif buffer is not None:
            gap = seg["start"] - buffer["end"]
            spacer = " " if gap < 2.0 else ". "
            merged = {
                "start": buffer["start"],
                "end": seg["end"],
                "text": buffer["text"].rstrip() + spacer + seg["text"],
            }
            result.append(merged)
            buffer = None
        else:
            result.append(seg)

    if buffer is not None:
        result.append(buffer)

    return result
