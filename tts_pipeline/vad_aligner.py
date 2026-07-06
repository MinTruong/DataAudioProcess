"""VAD interval grouping and text alignment."""
import re

from tts_pipeline.punctuator import restore_punctuation


def group_vad_intervals(
    intervals: list[dict],
    min_dur: float = 5.0,
    max_dur: float = 20.0,
) -> list[dict]:
    """Gop speech intervals thanh group 5-20s.

    Greedy: gom interval cho den ``max_dur``, emit group, reset.
    Tail luon duoc emit.
    """
    if not intervals:
        return []

    groups: list[list[dict]] = []
    current = [intervals[0]]

    for iv in intervals[1:]:
        group_dur = current[-1]["end"] - current[0]["start"]
        iv_dur = iv["end"] - iv["start"]

        if group_dur < min_dur:
            current.append(iv)
        elif group_dur + iv_dur > max_dur and group_dur >= min_dur:
            groups.append(current)
            current = [iv]
        else:
            current.append(iv)

    groups.append(current)  # always emit tail

    return [
        {"start": g[0]["start"], "end": g[-1]["end"]}
        for g in groups
    ]


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Fraction of (a) that overlaps with (b)."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    duration = a_end - a_start
    if duration <= 0:
        return 0.0
    return overlap / duration


def align_text_to_groups(
    groups: list[dict],
    raw_vtt_cues: list[dict],
    punctuation: bool = True,
) -> list[dict]:
    """Gan text cho moi VAD group tu raw VTT cues overlap.

    Voi moi group:
    1. Tim raw VTT cues co overlap > 0.3 voi group
    2. Lay text, join, clean whitespace
    3. Neu punctuation=True, chay restore_punctuation()
    4. Return {start, end, text}
    """
    results = []
    for g in groups:
        texts = []
        for cue in raw_vtt_cues:
            ratio = _overlap_ratio(
                cue["start"], cue["end"],
                g["start"], g["end"],
            )
            if ratio > 0.3:
                texts.append(cue["text"])

        joined = " ".join(texts)
        joined = re.sub(r"\s+", " ", joined).strip()

        if punctuation and joined:
            joined = restore_punctuation(joined)

        results.append({
            "start": g["start"],
            "end": g["end"],
            "text": joined,
        })

    return results
