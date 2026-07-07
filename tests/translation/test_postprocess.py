from __future__ import annotations

from src.llm.postprocess import (
    normalize_answer_text,
    deduplicate_citations,
    enforce_fallback_if_empty,
)


def test_normalize_answer_text():
    text = "Hello   world\n\n\nThis is text."
    cleaned = normalize_answer_text(text)
    assert cleaned == "Hello world\n\nThis is text."


def test_deduplicate_citations():
    text = "Rule applies [1]. Another line [1]."
    cleaned = deduplicate_citations(text)
    assert cleaned.count("[1]") == 2  # preserves inline use, no destructive cleanup


def test_enforce_fallback_if_empty():
    assert "couldn't find relevant information" in enforce_fallback_if_empty("")