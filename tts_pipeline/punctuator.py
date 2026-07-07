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

# Connector words that typically belong to the NEXT sentence but get
# stuck at the end of a segment due to VTT word-level cue overlap.
_TAIL_JUNK_RE = re.compile(
    r"\b(nhưng mà|cho nên|bởi vì|tại vì|với lại|rồi thì|thế là|vậy là)[.!?,\s]*$",
    re.IGNORECASE,
)


def _strip_edge_junk(segments: list[dict]) -> list[dict]:
    """Strip residual connector words from segment edges (VTT overlap).

    YouTube word-level cues often include the first word(s) of the next
    utterance at the end of the current cue. After merge_cues, these
    connector words sit at the end of the segment, causing audio-text
    mismatch.

    Also strips the same connector from the START of the next segment
    if it is a known connector, since dedup will handle longer overlaps.
    """
    result = []
    for seg in segments:
        text = seg["text"]
        if text:
            # Strip trailing connector junk
            text = _TAIL_JUNK_RE.sub("", text).strip()
            # Strip leading connector junk (residual from prev segment)
            text = re.sub(
                r"^(nhưng mà|cho nên|bởi vì|tại vì|với lại|rồi thì|thế là|vậy là)\s+",
                "", text, flags=re.IGNORECASE,
            ).strip()
        result.append({**seg, "text": text})
    return result


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


def _split_segment(s: dict) -> list[dict]:
    """Restore punctuation on text only, keeping original segment timing intact."""
    text = s["text"] or ""
    if not text.strip():
        return [s]
    text = restore_punctuation(text)
    # Strip residual connector words from VTT cue overlap
    text = _TAIL_JUNK_RE.sub("", text).strip()
    s["text"] = text
    return [s]


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
