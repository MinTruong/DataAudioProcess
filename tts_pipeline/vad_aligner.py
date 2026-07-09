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
        new_group_dur = iv["end"] - current[0]["start"]

        if new_group_dur > max_dur and group_dur >= min_dur:
            groups.append(current)
            current = [iv]
        elif new_group_dur > max_dur and group_dur < min_dur:
            groups.append(current)
            current = [iv]
        else:
            current.append(iv)

    groups.append(current)

    return [
        {"start": g[0]["start"], "end": g[-1]["end"]}
        for g in groups
    ]


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    duration = a_end - a_start
    if duration <= 0:
        return 0.0
    return overlap / duration


def _dedup_consecutive_text(texts: list[str]) -> list[str]:
    """Dedup overlapping word-level VTT cues.

    YouTube word-level cues are incremental: each cue extends the
    previous one with suffix overlap. Rules:
    1. Current is substring of last added -> skip
    2. Current extends last added (last is prefix) -> replace last
    3. Last added ends with current -> skip
    4. Current overlaps suffix of last added -> append only non-overlap
    5. Otherwise -> append as-is
    """
    if not texts:
        return []

    result = [texts[0]]
    for t in texts[1:]:
        if not t:
            continue
        prev = result[-1]

        if t in prev:
            continue
        if t.startswith(prev):
            result[-1] = t
            continue
        if prev.endswith(t):
            continue

        overlap_len = 0
        max_check = min(len(prev), len(t))
        for i in range(max_check, 0, -1):
            if prev[-i:] == t[:i]:
                overlap_len = i
                break

        if overlap_len > 0:
            result.append(t[overlap_len:])
        else:
            result.append(t)

    return result


def align_text_to_groups(
    groups: list[dict],
    raw_vtt_cues: list[dict],
    punctuation: bool = True,
) -> list[dict]:
    """Gan text cho moi VAD group tu raw VTT cues overlap.

    Moi VTT cue chi duoc gan vao 1 group duy nhat (best overlap ratio).
    Word-level cues duoc dedup. Text overlap giua cac segment duoc trim.

    Khong expand group timing — giu nguyen VAD boundary.
    """
    results = []
    if not groups or not raw_vtt_cues:
        return results

    _strip = lambda t: re.sub(r"<[^>]+>", "", t).strip()

    # Expand first/last edge for full coverage
    groups_adj = list(groups)
    if groups_adj:
        groups_adj[0]["start"] = min(
            groups_adj[0]["start"], raw_vtt_cues[0]["start"]
        )
        groups_adj[-1]["end"] = max(
            groups_adj[-1]["end"], raw_vtt_cues[-1]["end"]
        )

    # Pass 1: assign each VTT cue to the group it MOST overlaps
    group_cues: dict[int, list[dict]] = {i: [] for i in range(len(groups_adj))}
    for cue in raw_vtt_cues:
        text = _strip(cue["text"])
        if not text:
            continue
        best_idx = None
        best_ratio = 0.0
        for i, g in enumerate(groups_adj):
            r = _overlap_ratio(cue["start"], cue["end"], g["start"], g["end"])
            if r > best_ratio:
                best_ratio = r
                best_idx = i
        if best_idx is not None:
            group_cues[best_idx].append(cue)

    # Pass 2: expand group start backward to cover the earliest VTT cue
    # assigned to it. This fixes audio cutoff when VAD misses speech onset.
    # Only expand START, never end — expanding end causes cascade.
    for g_idx in range(1, len(groups_adj)):
        cues = group_cues[g_idx]
        if not cues:
            continue
        earliest_start = min(c["start"] for c in cues)
        if earliest_start < groups_adj[g_idx]["start"]:
            prev_end = groups[g_idx - 1]["end"]
            new_start = max(earliest_start, prev_end + 0.05)
            groups_adj[g_idx]["start"] = new_start

    # Pass 3: build preliminary segments (no punctuation yet)
    raw_texts = []
    for g_idx, g in enumerate(groups_adj):
        texts = [_strip(c["text"]) for c in group_cues[g_idx]]
        texts = _dedup_consecutive_text(texts)
        joined = " ".join(texts)
        joined = re.sub(r"\s+", " ", joined).strip()
        raw_texts.append({
            "start": g["start"],
            "end": g["end"],
            "text": joined,
        })

    # Trim text overlap between adjacent segments
    for i in range(1, len(raw_texts)):
        prev_text = raw_texts[i - 1]["text"]
        curr_text = raw_texts[i]["text"]
        if not prev_text or not curr_text:
            continue
        p_words = prev_text.split()
        c_words = curr_text.split()
        max_overlap = min(5, len(c_words))
        overlap_count = 0
        for n in range(max_overlap, 0, -1):
            head = " ".join(c_words[:n]).lower()
            tail = " ".join(p_words[-n:]).lower()
            if head in tail or tail in head:
                overlap_count = n
                break
        if overlap_count > 0:
            raw_texts[i]["text"] = " ".join(c_words[overlap_count:])

    # Pass 4: punctuate after overlap trimming
    for seg in raw_texts:
        text = seg["text"]
        if punctuation and text:
            text = restore_punctuation(text)
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
        })

    return results
