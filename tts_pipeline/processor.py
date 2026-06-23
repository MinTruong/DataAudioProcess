"""Text cleaning, deduplication, and segment splitting."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def dedup_consecutive_text(segments: list[dict]) -> list[dict]:
    """Remove overlapping text between consecutive segments.

    YouTube captions cause the last 1-3 words of segment N to appear
    at the start of segment N+1. This strips the overlap.
    """
    result: list[dict] = []
    for seg in segments:
        text = seg["text"]

        if result:
            prev_text = result[-1]["text"]
            prev_words = prev_text.split()

            # Check 3-word overlap
            if len(prev_words) >= 3:
                candidate = " ".join(prev_words[-3:])
                if text.startswith(candidate):
                    text = text[len(candidate):].strip().lstrip(",").strip()
                    if not text:
                        continue
                    text = text[0].upper() + text[1:] if text else text

            # Check 2-word overlap
            if text and result and len(prev_words) >= 2:
                candidate = " ".join(prev_words[-2:])
                if text.startswith(candidate):
                    text = text[len(candidate):].strip().lstrip(",").strip()
                    if not text:
                        continue
                    text = text[0].upper() + text[1:] if text else text

            # Check 1-word overlap
            if text and result:
                last_word = prev_words[-1] if prev_words else ""
                if last_word and len(last_word) > 1 and text.startswith(last_word):
                    text = text[len(last_word):].strip().lstrip(",").strip()
                    if not text:
                        continue
                    text = text[0].upper() + text[1:] if text else text

        result.append({**seg, "text": text})

    return result


def split_long_segments(
    segments: list[dict], max_dur: float = 20.0, min_dur: float = 2.0
) -> list[dict]:
    """Split segments longer than max_dur by sentence boundary.

    Time is prorated by character count across sentences.
    """
    result: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg["text"]

        if dur <= max_dur:
            result.append(seg)
            continue

        # Try sentence split, fallback to comma split
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
