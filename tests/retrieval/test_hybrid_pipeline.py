from __future__ import annotations

from src.retrieval.hybrid import merge_hybrid_results
from src.retrieval.pipeline import calculate_confidence


def test_merge_hybrid_results_combines_sources():
    dense = [{"chunk_id": "c1", "text": "A", "dense_score": 0.9}]
    sparse = [{"chunk_id": "c1", "text": "A", "sparse_score": 12.0}]
    merged = merge_hybrid_results(dense, sparse, top_k=5)
    assert len(merged) == 1
    assert merged[0]["chunk_id"] == "c1"


def test_calculate_confidence_high():
    candidates = [
        {"dense_score": 0.88, "rerank_score": 8.7},
        {"dense_score": 0.84, "rerank_score": 8.1},
    ]
    score, label = calculate_confidence(candidates)
    assert 0 <= score <= 1
    assert label in {"High", "Medium", "Low"}