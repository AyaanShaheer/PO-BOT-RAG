from __future__ import annotations

import numpy as np

from src.retrieval.faiss_store import (
    build_faiss_index,
    search_faiss_index,
)


def test_build_faiss_index():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    index = build_faiss_index(embeddings)
    assert index.ntotal == 3


def test_search_faiss_index_returns_ranked_ids():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    index = build_faiss_index(embeddings)
    query = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    scores, indices = search_faiss_index(index, query, top_k=2)
    assert indices.shape == (1, 2)
    assert indices[0][0] == 0