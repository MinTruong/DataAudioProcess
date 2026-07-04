"""Vietnamese punctuation restoration for raw caption text."""

import re
from pathlib import Path

from underthesea import sent_tokenize

from tts_pipeline.parser import export_segments_to_vtt, load_vtt_segments

_QUESTION_WORDS = {
    "ai", "gì", "đâu", "sao", "nào", "bao nhiêu",
    "tại sao", "vì sao", "thế nào", "ra sao",
    "à", "hả", "nhỉ", "nhé", "chứ", "chăng", "ư", "hử", "hở",
}

_CLAUSE_SPLIT_MARKERS = [
    "nhưng", "cho nên là", "cho nên", "tại vì", "bởi vì",
    "rồi thì", "thế là", "vậy là", "tuy nhiên", "với lại",
    "nói chung là", "nói chung",
]


def _split_by_clause(text: str, min_clause_len: int = 40) -> list[str]:
    """Split Vietnamese run-on text into clause chunks.

    Uses discourse markers that reliably start a new clause in spoken
    Vietnamese (nhưng, cho nên, tại vì, rồi thì, thế là, etc.).

    Only splits when the preceding clause is at least ``min_clause_len``
    characters, preventing degenerate short clauses.

    Returns:
        list of clause strings, or ``[text.strip()]`` if no split possible.
    """
    if not text or not text.strip():
        return [text] if text else []

    # Longer markers first so multi-word markers match before their subwords
    markers = sorted(_CLAUSE_SPLIT_MARKERS, key=len, reverse=True)

    # Pattern: position after whitespace that's followed by a clause marker
    pattern = r'(?<=\s)(?=(?:' + '|'.join(re.escape(m) for m in markers) + r')\b)'

    parts = []
    prev = 0
    for m in re.finditer(pattern, text):
        pos = m.start()
        if pos - prev >= min_clause_len:
            chunk = text[prev:pos].strip()
            if chunk:
                parts.append(chunk)
            prev = pos

    tail = text[prev:].strip()
    if tail:
        parts.append(tail)

    return parts if parts else [text.strip()]


_EXCLAMATION_STARTS = {
    "trời", "chao", "ôi", "á", "úi", "ối", "eo",
    "chà", "ồ", "ô hay", "ôi trời", "trời ơi", "trời đất",
}


def _detect_punctuation(sentence: str) -> str:
    """Detect end punctuation for a sentence.

    Returns one of '.', '?', or '!'.
    """
    lower = sentence.lower().strip()
    if not lower:
        return "."

    # Exclamation if sentence starts with an exclamation word
    for start_word in _EXCLAMATION_STARTS:
        if lower.startswith(start_word):
            return "!"

    # Question if sentence contains question words
    words = set(re.sub(r"[^\w\s]", " ", lower).split())
    if words & _QUESTION_WORDS:
        return "?"

    return "."


def _punctuate_sentence(sent: str) -> str:
    """Add end punctuation and capitalize a single sentence."""
    sent = sent.strip()
    if not sent:
        return ""

    sent = sent.rstrip(".!?,")
    punct = _detect_punctuation(sent)
    if sent and sent[0].islower():
        sent = sent[0].upper() + sent[1:]
    return sent + punct


def _pre_split_text(text: str) -> list[str]:
    """Split text into segments suitable for sentence tokenization.

    If ``text`` contains no ``. ! ?`` punctuation and exceeds 120
    characters, run clause splitting first to give ``sent_tokenize``
    manageable chunks.
    """
    if not text or not text.strip():
        return [text] if text else []

    _has_explicit_punct = any(c in text for c in ".!?")
    if _has_explicit_punct or len(text) <= 120:
        return [text]

    return _split_by_clause(text)


def restore_punctuation(text: str) -> str:
    """Add end punctuation (., ? , !) and capitalize sentences in raw text.

    Uses underthesea.sent_tokenize() for Vietnamese-aware sentence
    segmentation, then appends a period, question mark, or exclamation
    mark based on content heuristics.

    For long text without any punctuation, pre-splits by clause markers
    (nhưng, cho nên, rồi thì, etc.) so that sentence tokenization
    produces reasonable boundaries.

    Args:
        text: Raw text without punctuation.

    Returns:
        Text with punctuation added and sentences capitalized.
    """
    if not text or not text.strip():
        return text

    chunks = _pre_split_text(text)

    result = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        sentences = sent_tokenize(chunk)
        if not sentences:
            result.append(_punctuate_sentence(chunk))
            continue
        for sent in sentences:
            punct_sent = _punctuate_sentence(sent)
            if punct_sent:
                result.append(punct_sent)

    return " ".join(result)


def _split_segment(s: dict, min_len: int = 150) -> list[dict]:
    """Split one segment by clause markers if it has no punctuation.

    Each clause gets a prorated time slice based on character ratio.
    If the segment already has punctuation or is short, return as-is
    (with punctuation restored).
    """
    text = s["text"] or ""
    if not text.strip():
        return [s]

    # Always restore punctuation on the original text first
    text = restore_punctuation(text)

    # Only split if text has no sentence punctuation and is long enough
    _has_punct = any(c in text for c in ".!?")
    if _has_punct and len(text) < min_len:
        s["text"] = text
        return [s]

    # If text has punctuation, try sentence split
    if _has_punct:
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p and p.strip()]
    else:
        # No punctuation → use clause split
        chunks = _split_by_clause(text)
        if len(chunks) <= 1:
            s["text"] = text
            return [s]
        # Punctuate each chunk then split
        parts = []
        for ch in chunks:
            punct_ch = restore_punctuation(ch)
            sub = re.split(r"(?<=[.!?])\s+", punct_ch)
            parts.extend(p.strip() for p in sub if p and p.strip())

    if len(parts) <= 1:
        s["text"] = text
        return [s]

    # Filter out degenerate tail clauses: short text (< 3 words) that is
    # a suffix of the previous clause (VTT overlap artifact).
    filtered = [parts[0]]
    for i in range(1, len(parts)):
        prev = filtered[-1]
        curr = parts[i]
        prev_words_5 = " ".join(prev.split()[-5:]).lower()
        curr_lower = curr.lower().strip()
        # If current is <= 3 words and is contained in prev's tail → skip
        if len(curr.split()) <= 3 and curr_lower in prev_words_5:
            continue
        filtered.append(curr)

    if len(filtered) <= 1:
        s["text"] = text
        return [s]
    parts = filtered

    # Prorate time within THIS segment only
    dur = s["end"] - s["start"]
    total_chars = max(sum(len(p) for p in parts), 1)
    char_ratio = dur / total_chars

    out = []
    cur = s["start"]
    for part in parts:
        part_dur = max(len(part) * char_ratio, 0.3)  # min 0.3s
        out.append({
            "start": cur,
            "end": min(cur + part_dur, s["end"]),
            "text": part,
        })
        cur += part_dur * 1.0  # sequential, no gap

    # Clamp last end
    if out:
        out[-1]["end"] = s["end"]

    return out


def punctuate_vtt_file(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict]:
    """Read a VTT file, restore punctuation, split by .!?, write a new VTT.

    Operates per-segment: each merged segment is punctuated and possibly
    split independently. This preserves the original VTT timing boundaries
    and avoids global time proration errors.

    Returns:
        list of {start, end, text}
    """
    segments = load_vtt_segments(input_path)

    if not segments:
        export_segments_to_vtt([], output_path)
        return []

    output = []
    for s in segments:
        split = _split_segment(s)
        output.extend(split)

    # Remove segments that are short (<=3 words) and whose text is a
    # suffix of the previous segment — VTT overlap artifacts.
    filtered = []
    for i, seg in enumerate(output):
        curr_words = seg["text"].split()
        if len(curr_words) <= 3 and i > 0:
            prev_text = output[i - 1]["text"]
            curr_raw = seg["text"].lower().strip(".!?,")
            # Check if current text is a suffix of prev's last 10 words
            prev_tail = " ".join(prev_text.lower().split()[-10:])
            if curr_raw in prev_tail and len(curr_raw) > 1:
                continue  # skip — it's a redundant tail
        filtered.append(seg)
    output = filtered

    export_segments_to_vtt(output, output_path)
    return output
