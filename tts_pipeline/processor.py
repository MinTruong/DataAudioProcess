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


def _strip_music_noise(text: str) -> str:
    """Remove music/noise/copyright artifacts from text, preserving remaining content."""
    result = text
    for pattern in _MUSIC_PATTERNS:
        result = result.replace(pattern, "")
    result = result.strip()
    return result


def _strip_music_noise_from_segment(seg: dict) -> dict | None:
    """Return segment with music/noise stripped, or None if entirely removed."""
    if not seg["text"]:
        return None
    if not _has_music_noise(seg["text"]):
        return seg
    cleaned = _strip_music_noise(seg["text"])
    if not cleaned:
        return None
    return {**seg, "text": cleaned}


def segment_by_content(
    segments: list[dict],
    min_dur: float = 5.0,
    max_dur: float = 20.0,
    **kwargs,
) -> list[dict]:
    """Group merged segments into 5-20s groups by duration.

    Each input segment is treated as an atomic unit (no intra-segment
    splitting via punctuation/character-ratio). This prevents audio bleed
    caused by YouTube VTT cue overlap when prorating time per sentence.

    Music/noise artifacts are stripped per segment (not dropped entirely).
    Tail group is always emitted even if short.
    """
    # Filter/strip music/noise
    clean: list[dict] = []
    for seg in segments:
        result = _strip_music_noise_from_segment(seg)
        if result is not None:
            clean.append(result)
    if not clean:
        return []

    # Greedy duration-based grouping on whole segments
    groups: list[list[dict]] = []
    current = [clean[0]]

    for seg in clean[1:]:
        group_dur = current[-1]["end"] - current[0]["start"]
        seg_dur = seg["end"] - seg["start"]

        if group_dur < min_dur:
            current.append(seg)
        elif group_dur + seg_dur > max_dur and group_dur >= min_dur:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)

    groups.append(current)  # always emit tail

    output = []
    for group in groups:
        texts = [s["text"] for s in group]
        cleaned = clean_text(" ".join(texts))
        start = group[0]["start"]
        end = group[-1]["end"]
        output.append({"start": start, "end": end, "text": cleaned})

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
            # Iterate all match positions to find a valid word-boundary match
            limit = min(len(prev_words), 15)
            for n in range(limit, 0, -1):
                prev_suffix = " ".join(prev_words[-n:])
                search_start = 0
                found = False
                while True:
                    pos = curr_lower.find(prev_suffix.lower(), search_start)
                    if pos == -1:
                        break  # no more matches at this n level
                    if pos == 0 or curr_lower[pos - 1].isspace():
                        # Valid word-boundary match — use it
                        new_text = text[pos + len(prev_suffix):].strip()
                        new_text = new_text.lstrip(",").strip()
                        if new_text:
                            text = new_text[0].upper() + new_text[1:]
                            found = True
                        break  # either accepted or empty — try shorter suffix
                    search_start = pos + 1  # try next position
                if found:
                    break  # valid match found and processed
                # else continue to shorter suffix

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
