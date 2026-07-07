from __future__ import annotations

import re

from config.prompts import FALLBACK_RESPONSE


def normalize_answer_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate_citations(text: str) -> str:
    """
    Conservative cleanup only.
    We do NOT remove repeated citation references across sentences,
    because that can silently break grounding clarity.
    """
    return re.sub(r"\[\s+(\d+)\s+\]", r"[\1]", text)


def enforce_fallback_if_empty(text: str) -> str:
    cleaned = normalize_answer_text(text)
    if not cleaned:
        return FALLBACK_RESPONSE
    return cleaned


def postprocess_answer(text: str) -> str:
    text = normalize_answer_text(text)
    text = deduplicate_citations(text)
    text = enforce_fallback_if_empty(text)
    return text