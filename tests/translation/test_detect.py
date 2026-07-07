from __future__ import annotations

from src.translation.detect import detect_query_language, normalize_language_code


def test_normalize_language_code_english():
    assert normalize_language_code("en") == "en"


def test_normalize_language_code_tagalog():
    assert normalize_language_code("tl") == "tl"
    assert normalize_language_code("fil") == "tl"


def test_detect_query_language_english_fallback():
    text = "What are annual leave rules in Hong Kong?"
    result = detect_query_language(text)
    assert result in {"en", "tl"}