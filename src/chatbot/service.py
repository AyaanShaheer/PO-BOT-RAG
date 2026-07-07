from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable

from config.prompts import FALLBACK_RESPONSE
from src.llm.generator import generate_answer
from src.llm.postprocess import postprocess_answer
from src.retrieval.pipeline import QueryResult, run_query_pipeline
from src.translation.translate import translate_from_english, translate_to_english
from utils.metrics import LatencyTracker, Timer


@dataclass(slots=True)
class ChatResponse:
    answer_text: str
    displayed_answer: str
    query_result: QueryResult
    generation_latency: str
    translated_back: bool


def stream_answer_tokens(
    question: str,
    context_chunks: list[dict],
) -> Generator[str, None, None]:
    stream = generate_answer(
        question=question,
        context_chunks=context_chunks,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            delta = None
        if delta:
            yield delta


def format_final_answer(
    answer_text: str,
    confidence_score: float,
    confidence_label: str,
    sources: list[dict],
    retrieval_latency: str,
    generation_latency: str,
) -> str:
    unique_sources: list[str] = []
    seen: set[tuple[str, str]] = set()

    for item in sources:
        title = item.get("title", "Unknown")
        page = str(item.get("page", "unknown"))
        source = item.get("source", "Unknown")
        key = (title, page)
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(f"- {title} — Page {page} — {source}")

    sources_block = "\n".join(unique_sources) if unique_sources else "- No sources"

    return (
        f"{answer_text}\n\n"
        f"Confidence: {confidence_score:.2f} ({confidence_label})\n"
        f"Retrieval Latency: {retrieval_latency}\n"
        f"Generation Latency: {generation_latency}\n"
        f"Sources:\n{sources_block}"
    )


def run_chat_request(question: str) -> ChatResponse:
    tracker = LatencyTracker()
    query_result = run_query_pipeline(question)

    with Timer("generation") as t_generation:
        if query_result.fallback_used:
            answer_text = FALLBACK_RESPONSE
        else:
            answer_text = generate_answer(
                question=query_result.normalized_query,
                context_chunks=query_result.retrieved_chunks,
                stream=False,
            )

    tracker.record("generation", t_generation.elapsed_sec)
    answer_text = postprocess_answer(answer_text)

    translated_back = False
    displayed_answer = answer_text
    if query_result.detected_language != "en":
        displayed_answer = translate_from_english(answer_text, query_result.detected_language)
        translated_back = True

    return ChatResponse(
        answer_text=answer_text,
        displayed_answer=displayed_answer,
        query_result=query_result,
        generation_latency=tracker.summary().get("generation", "N/A"),
        translated_back=translated_back,
    )