from __future__ import annotations

from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from src.embedding.embed import get_embedding_model
from src.retrieval.bm25_store import search_bm25_index
from src.retrieval.faiss_store import search_faiss_index


def dense_search(
    query: str,
    index,
    metadata: list[dict],
    model: SentenceTransformer | None = None,
    top_k: int | None = None,
) -> list[dict]:
    settings = get_settings()
    top_k = top_k or settings.hybrid_top_k
    model = model or get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = search_faiss_index(index, query_embedding, top_k=top_k)

    results: list[dict] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        record = dict(metadata[idx])
        record["dense_score"] = float(score)
        results.append(record)
    return results


def sparse_search(
    query: str,
    bm25,
    metadata: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    settings = get_settings()
    top_k = top_k or settings.hybrid_top_k

    ranked = search_bm25_index(bm25, query, top_k=top_k)
    results: list[dict] = []
    for idx, score in ranked:
        record = dict(metadata[idx])
        record["sparse_score"] = float(score)
        results.append(record)
    return results


def merge_hybrid_results(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    settings = get_settings()
    top_k = top_k or settings.hybrid_top_k

    merged: dict[str, dict] = {}
    score_map = defaultdict(float)

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        merged[chunk_id] = dict(item)
        score_map[chunk_id] += 1.0 / rank

    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item["chunk_id"]
        if chunk_id not in merged:
            merged[chunk_id] = dict(item)
        else:
            merged[chunk_id].update(item)
        score_map[chunk_id] += 1.0 / rank

    ranked = sorted(
        merged.values(),
        key=lambda x: score_map[x["chunk_id"]],
        reverse=True,
    )
    return ranked[:top_k]