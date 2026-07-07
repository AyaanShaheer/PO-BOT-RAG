from __future__ import annotations

from groq import Groq

from config.config import get_groq_api_key
from config.prompts import FALLBACK_RESPONSE, build_rag_prompt
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def get_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())


def build_citation_block(chunks: list[dict]) -> str:
    lines: list[str] = []
    seen: set[tuple[str, str | None]] = set()

    for chunk in chunks:
        title = chunk.get("title", "Unknown Document")
        page = chunk.get("page")
        source = chunk.get("source", "Unknown Source")
        key = (title, str(page))

        if key in seen:
            continue
        seen.add(key)

        page_label = f"Page {page}" if page not in (None, "", "?") else "Page unknown"
        lines.append(f"- {title} — {page_label} — {source}")

    return "\n".join(lines)


def generate_answer(
    question: str,
    context_chunks: list[dict],
    stream: bool = False,
):
    settings = get_settings()

    if not context_chunks:
        return FALLBACK_RESPONSE

    prompt = build_rag_prompt(context_chunks, question)
    client = get_groq_client()

    messages = [
        {"role": "system", "content": "You are a grounded legal RAG assistant."},
        {"role": "user", "content": prompt},
    ]

    if stream:
        return client.chat.completions.create(
            model=settings.llm_model_name,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=700,
            stream=True,
        )

    response = client.chat.completions.create(
        model=settings.llm_model_name,
        messages=messages,
        temperature=0.1,
        max_completion_tokens=700,
        stream=False,
    )
    content = response.choices[0].message.content or FALLBACK_RESPONSE
    return content.strip()