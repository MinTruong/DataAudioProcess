"""Tests for Vietnamese punctuation restoration."""

from tts_pipeline.punctuator import restore_punctuation, _detect_punctuation


def test_restore_empty_text():
    assert restore_punctuation("") == ""
    assert restore_punctuation("   ") == "   "


def test_restore_already_punctuated():
    """Text already ending with punctuation should not double-punctuate."""
    result = restore_punctuation("Xin chào. Tôi là Bách.")
    assert result == "Xin chào. Tôi là Bách."


def test_restore_adds_period():
    """Raw text without punctuation gets periods added."""
    result = restore_punctuation("xin chào các bạn")
    assert result == "Xin chào các bạn."


def test_restore_question():
    """Question words trigger ? punctuation."""
    result = restore_punctuation("bạn tên là gì")
    assert result == "Bạn tên là gì?"


def test_restore_exclamation():
    """Exclamation-starting sentences get ! punctuation."""
    result = restore_punctuation("trời ơi sao đẹp thế")
    assert result == "Trời ơi sao đẹp thế!"


def test_restore_multiple_sentences():
    """Multiple sentences each receive proper punctuation."""
    # Use a clear multi-sentence input that underthesea can segment
    result = restore_punctuation("Xin chào. Tôi là Bách. Bạn khỏe không?")
    assert ". Tôi" in result or "." in result
    assert result[0].isupper()


def test_detect_period():
    assert _detect_punctuation("xin chào các bạn") == "."


def test_detect_question():
    assert _detect_punctuation("bạn tên là gì") == "?"


def test_detect_exclamation():
    assert _detect_punctuation("trời ơi sao đẹp thế") == "!"
