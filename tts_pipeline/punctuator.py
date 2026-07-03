"""Vietnamese punctuation restoration for raw caption text."""

import re
from underthesea import sent_tokenize

_QUESTION_WORDS = {
    "ai", "gì", "đâu", "sao", "nào", "bao nhiêu",
    "tại sao", "vì sao", "thế nào", "ra sao",
    "à", "hả", "nhỉ", "nhé", "chứ", "chăng", "ư", "hử", "hở",
}

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


def restore_punctuation(text: str) -> str:
    """Add end punctuation (., ? , !) and capitalize sentences in raw text.

    Uses underthesea.sent_tokenize() for Vietnamese-aware sentence
    segmentation, then appends a period, question mark, or exclamation
    mark based on content heuristics.

    Args:
        text: Raw text without punctuation.

    Returns:
        Text with punctuation added and sentences capitalized.
    """
    if not text or not text.strip():
        return text

    sentences = sent_tokenize(text)
    if not sentences:
        return text

    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Strip existing trailing punctuation to avoid doubling
        sent = sent.rstrip(".!?,")

        punct = _detect_punctuation(sent)

        # Capitalize first letter
        if sent[0].islower():
            sent = sent[0].upper() + sent[1:]

        result.append(sent + punct)

    return " ".join(result)
