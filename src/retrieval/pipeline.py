from __future__ import annotations

from dataclasses import dataclass

from config.prompts import FALLBACK_RESPONSE
from config.settings import get_settings
from src.retrieval.bm25_store import load_bm25_index
from src.retrieval.faiss_store import load_faiss_index
from src.retrieval.hybrid import dense_search, merge_hybrid_results, sparse_search
from src.retrieval.query_expander import expand_query
from src.retrieval.reranker import rerank_results
from src.translation.detect import detect_query_language
from src.translation.translate import translate_from_english, translate_to_english
from utils.logger import get_logger
from utils.metrics import LatencyTracker, Timer

logger = get_logger(__name__)


@dataclass(slots=True)
class QueryResult:
    original_query: str
    normalized_query: str
    detected_language: str
    expanded_queries: list[str]
    retrieved_chunks: list[dict]
    answer: str | None = None
    confidence_score: float = 0.0
    confidence_label: str = "Low"
    latency: dict[str, str] | None = None
    fallback_used: bool = False


def calculate_confidence(candidates: list[dict]) -> tuple[float, str]:
    if not candidates:
        return 0.0, "Low"

    top_dense = max(float(c.get("dense_score", 0.0)) for c in candidates)
    top_rerank = max(float(c.get("rerank_score", 0.0)) for c in candidates)

    dense_component = min(max(top_dense, 0.0), 1.0)
    rerank_component = min(max((top_rerank / 10.0), 0.0), 1.0)

    score = round((dense_component * 0.55) + (rerank_component * 0.45), 4)

    if score >= 0.75:
        label = "High"
    elif score >= 0.45:
        label = "Medium"
    else:
        label = "Low"

    return score, label


def retrieve_context(query: str) -> list[dict]:
    settings = get_settings()
    faiss_index, faiss_metadata = load_faiss_index()
    bm25, bm25_metadata = load_bm25_index()

    expanded = expand_query(query)
    dense_results_all: list[dict] = []
    sparse_results_all: list[dict] = []

    for expanded_query in expanded:
        dense_results_all.extend(
            dense_search(
                query=expanded_query,
                index=faiss_index,
                metadata=faiss_metadata,
                top_k=settings.hybrid_top_k,
            )
        )
        sparse_results_all.extend(
            sparse_search(
                query=expanded_query,
                bm25=bm25,
                metadata=bm25_metadata,
                top_k=settings.hybrid_top_k,
            )
        )

    merged = merge_hybrid_results(
        dense_results=dense_results_all,
        sparse_results=sparse_results_all,
        top_k=settings.hybrid_top_k,
    )
    reranked = rerank_results(
        query=query,
        candidates=merged,
        top_k=settings.rerank_top_k,
    )
    return reranked


def run_query_pipeline(query: str) -> QueryResult:
    tracker = LatencyTracker()
    detected_language = detect_query_language(query)
    normalized_query = translate_to_english(query, detected_language)
    expanded_queries = expand_query(normalized_query)

    with Timer("retrieval") as t_retrieval:
        retrieved_chunks = retrieve_context(normalized_query)
    tracker.record("retrieval", t_retrieval.elapsed_sec)

    confidence_score, confidence_label = calculate_confidence(retrieved_chunks)

    fallback_used = len(retrieved_chunks) == 0
    answer = None

    return QueryResult(
        original_query=query,
        normalized_query=normalized_query,
        detected_language=detected_language,
        expanded_queries=expanded_queries,
        retrieved_chunks=retrieved_chunks,
        answer=answer,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        latency=tracker.summary(),
        fallback_used=fallback_used,
    )