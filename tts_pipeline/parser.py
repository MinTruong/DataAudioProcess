"""VTT caption parsing, cue merging, and VTT file I/O."""

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
    """Merge YouTube auto-caption fragments and split sentences into clean segments.

    YouTube VTT structure per utterance:
      word-level cue A (partial, 1-2s)
      plain fragment 0.01s (full text of A)
      word-level cue A' (partial, continuation of A)
      plain fragment 0.01s (full text of A + A')
      word-level cue B (partial, start of next utterance)
      ...

    This function:
    - Fragments (< 0.3s) ONLY extend time, NEVER overwrite text
    - Word-level cues that continue the same sentence are merged (text combined)
    - A new segment starts when the previous text ends with . ! ?
    - The longest suffix overlap between consecutive cues is deduplicated
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

        # Fragment (< threshold): extend last segment's end time ONLY
        if dur < s.fragment_threshold:
            if merged:
                merged[-1]["end"] = max(merged[-1]["end"], cue["end"])
            i += 1
            continue

        # Word-level cue
        curr_text = cue["text"]
        if not curr_text:
            i += 1
            continue

        if not merged:
            merged.append({"start": cue["start"], "end": cue["end"], "text": curr_text})
            i += 1
            continue

        prev = merged[-1]
        prev_text = prev["text"]
        prev_lower = prev_text.lower()
        curr_lower = curr_text.lower()

        # Case 1: current text extends prev (more complete version)
        if curr_lower.startswith(prev_lower) and len(curr_text) > len(prev_text):
            prev["text"] = curr_text
            prev["end"] = cue["end"]
            i += 1
            continue

        # Case 2: prev fully contains current → skip (just extend time)
        if prev_lower.startswith(curr_lower) and len(prev_text) >= len(curr_text):
            prev["end"] = max(prev["end"], cue["end"])
            i += 1
            continue

        # Force-break: if prev text is very long without sentence-ending
        # punctuation, assume a natural boundary exists. This handles
        # videos where auto-caption has no punctuation cues.
        if len(prev_text) > 250 and not any(c in prev_text[-150:] for c in ".!?"):
            merged.append({"start": cue["start"], "end": cue["end"], "text": curr_text})
            i += 1
            continue

        # Case 3: prev ends with sentence punctuation
        prev_ends = prev_text.rstrip()[-1] if prev_text.rstrip() else ""
        if prev_ends in ".!?":
            # If current text is purely a repeat of prev's tail → skip
            prev_words = prev_text.split()
            pure_overlap = False
            for n in range(min(len(prev_words), 20), 0, -1):
                suffix = " ".join(prev_words[-n:]).lower()
                if curr_lower.startswith(suffix):
                    stripped = curr_text[len(suffix):].strip().lstrip(",").strip()
                    if not stripped:
                        pure_overlap = True
                    break

            if pure_overlap:
                prev["end"] = cue["end"]
                i += 1
                continue

            # New segment — dedup in processor.py handles residual overlap
            merged.append({"start": cue["start"], "end": cue["end"], "text": curr_text})
            i += 1
            continue

        # Case 4: no sentence punctuation → merge into current sentence
        # (uppercase in Vietnamese can be a proper noun mid-sentence)

        # Case 5: continuation of same sentence → combine
        # Find longest suffix overlap between prev and current to avoid duplication
        prev_words = prev_text.split()
        overlap_n = 0
        limit = min(len(prev_words), 20)
        for n in range(limit, 0, -1):
            suffix = " ".join(prev_words[-n:]).lower()
            if curr_lower.startswith(suffix):
                overlap_n = n
                break

        rest = curr_text
        if overlap_n > 0:
            overlap = " ".join(prev_words[-overlap_n:])
            rest = curr_text[len(overlap):].strip()

        spacer = " " if rest else ""
        prev["text"] = prev_text + spacer + rest
        prev["end"] = cue["end"]
        i += 1

    return merged


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def export_segments_to_vtt(
    segments: list[dict], path: str | Path,
) -> None:
    """Export segment list to VTT file.

    Each segment becomes a VTT cue with start/end time and text.
    Useful as an intermediate artifact for inspection or re-processing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_vtt_time(seg["start"])
        end = _format_vtt_time(seg["end"])
        text = seg["text"]
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_vtt_segments(path: str | Path) -> list[dict]:
    """Load VTT file back into segment list [{start, end, text}, ...].

    Inverse of export_segments_to_vtt.
    """
    return parse_vtt_full(str(path))
