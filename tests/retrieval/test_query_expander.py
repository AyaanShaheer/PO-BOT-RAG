from __future__ import annotations

from src.retrieval.query_expander import expand_query


def test_expand_query_domestic_worker():
    expanded = expand_query("domestic worker rights")
    assert "domestic helper rights" in expanded
    assert "foreign domestic helper rights" in expanded


def test_expand_query_preserves_original():
    expanded = expand_query("annual leave")
    assert expanded[0] == "annual leave"