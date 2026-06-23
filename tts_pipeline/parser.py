"""VTT caption parsing and cue merging for YouTube auto-captions."""

import re
from pathlib import Path

from tts_pipeline.config import Settings


def parse_vtt_time(ts_str: str) -> float:
    """Convert VTT timestamp (00:00:01.560) to seconds."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _strip_vtt_markup(text: str) -> str:
    """Remove VTT word-level timestamps and markup tags."""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt_full(vtt_path: str | Path) -> list[dict]:
    """Parse YouTube VTT caption into raw time-aligned cues.

    Returns:
        list of {start: float, end: float, text: str}
    """
    raw_cues: list[dict] = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_start: float | None = None
    current_end: float | None = None
    cue_text: list[str] = []
    in_cue = False

    for line in lines:
        line = line.rstrip("\n")
        if " --> " in line:
            if current_start is not None and cue_text:
                raw_cues.append({
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(cue_text).strip(),
                })
            parts = line.split(" --> ")
            current_start = parse_vtt_time(parts[0])
            current_end = parse_vtt_time(parts[1].split()[0])
            cue_text = []
            in_cue = True
        elif in_cue:
            clean = line.strip()
            if clean and not clean.startswith("WEBVTT") and not clean.startswith("Kind:") and not clean.startswith("Language:"):
                cue_text.append(clean)

    if current_start is not None and cue_text:
        raw_cues.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(cue_text).strip(),
        })
    return raw_cues


def merge_cues(cues: list[dict], settings: Settings | None = None) -> list[dict]:
    """Merge YouTube auto-caption fragments into complete segments.

    YouTube generates 3 cues per utterance: word-level (long), plain fragment
    (0.01s), and the next utterance start. This merges them into clean segments.

    Args:
        cues: Raw cues from parse_vtt_full()
        settings: Pipeline Settings (controls merge_gap, fragment_threshold)

    Returns:
        list of {start: float, end: float, text: str} with markup stripped
    """
    s = settings or Settings()
    if not cues:
        return []

    for cue in cues:
        cue["text"] = _strip_vtt_markup(cue["text"])

    merged: list[dict] = []
    i = 0

    while i < len(cues):
        cue = cues[i]
        dur = cue["end"] - cue["start"]

        # Fragment (< threshold): merge time into previous segment
        if dur < s.fragment_threshold and merged:
            merged[-1]["end"] = max(merged[-1]["end"], cue["end"])
            if cue["text"] and len(cue["text"]) > len(merged[-1]["text"]):
                merged[-1]["text"] = cue["text"]
            i += 1
            continue

        combined_text = cue["text"]
        combined_end = cue["end"]
        j = i + 1

        while j < len(cues):
            next_dur = cues[j]["end"] - cues[j]["start"]
            next_text = cues[j]["text"]

            # Skip fragments
            if next_dur < s.fragment_threshold:
                if cues[j]["end"] > combined_end:
                    combined_end = cues[j]["end"]
                if next_text and len(next_text) > len(combined_text):
                    combined_text = next_text
                j += 1
                continue

            gap = cues[j]["start"] - combined_end
            if gap > s.merge_gap:
                break

            if next_text.startswith(combined_text) and len(next_text) > len(combined_text):
                combined_text = next_text
                combined_end = cues[j]["end"]
                j += 1
            elif next_text == combined_text:
                combined_end = cues[j]["end"]
                j += 1
            else:
                break

        merged.append({"start": cue["start"], "end": combined_end, "text": combined_text})
        i = j

    return merged
