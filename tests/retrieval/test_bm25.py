from __future__ import annotations

from src.retrieval.bm25_store import (
    tokenize_for_bm25,
    build_bm25_index,
    search_bm25_index,
)


def test_tokenize_for_bm25():
    text = "Section 57 annual leave entitlement"
    tokens = tokenize_for_bm25(text)
    assert "section" in tokens
    assert "57" in tokens


def test_bm25_search_returns_results():
    corpus = [
        "section 57 annual leave entitlement",
        "foreign domestic helper contract",
        "employment agency regulations",
    ]
    bm25 = build_bm25_index(corpus)
    ranked = search_bm25_index(bm25, "section 57", top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == 0