from __future__ import annotations

from sentence_transformers import CrossEncoder

from config.settings import get_settings


def get_reranker(model_name: str | None = None) -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(model_name or settings.reranker_model_name, max_length=512)


def rerank_results(
    query: str,
    candidates: list[dict],
    reranker: CrossEncoder | None = None,
    top_k: int | None = None,
) -> list[dict]:
    settings = get_settings()
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    reranker = reranker or get_reranker()
    pairs = [(query, item["text"]) for item in candidates]
    scores = reranker.predict(pairs)

    rescored: list[dict] = []
    for item, score in zip(candidates, scores):
        record = dict(item)
        record["rerank_score"] = float(score)
        rescored.append(record)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored[:top_k]