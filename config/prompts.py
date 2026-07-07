"""
prompts.py
----------
All prompt templates and instruction strings for the LLM layer.

Design philosophy:
  - Prompts are code. They live here, versioned, tested, not scattered in generator.py.
  - Each function returns a string. No side effects. Easy to unit-test.
  - Citation format matches FR-16. Fallback matches FR-14 reliability requirement.
"""

from __future__ import annotations

# ─── Constants ────────────────────────────────────────────────────────────────

FALLBACK_RESPONSE = (
    "I couldn't find relevant information in the available documents "
    "to answer your question. Please consult the official Hong Kong "
    "Labour Department website at https://www.labour.gov.hk for authoritative guidance."
)

CITATION_EXAMPLE = (
    "[1] Employment Ordinance (Cap. 57) — Page 17\n"
    "[2] Foreign Domestic Helper Guidelines — Page 8"
)


# ─── Prompt builders ──────────────────────────────────────────────────────────

def get_system_prompt() -> str:
    """
    Core system prompt injected before every LLM call.
    Enforces grounding, citation, and non-hallucination rules.
    """
    return (
        "You are PoBot, an expert AI assistant specialising in Hong Kong Labour Regulations.\n\n"
        "RULES — follow these without exception:\n"
        "1. Answer ONLY using information present in the CONTEXT provided below.\n"
        "2. Do NOT fabricate, invent, or extrapolate any legal information not in the context.\n"
        "3. If the context does not contain enough information to answer, output EXACTLY:\n"
        f'   "{FALLBACK_RESPONSE}"\n'
        "4. Every factual statement MUST be followed by a citation in the format:\n"
        "   [N] Document Title — Page X\n"
        "5. Number citations sequentially starting from [1].\n"
        "6. Be concise. Use plain English. Avoid legal jargon unless quoting directly.\n"
        "7. If the question is in Tagalog, answer in Tagalog (translation handled externally).\n"
    )


def get_citation_instructions() -> str:
    """Instructions injected into the prompt about citation format."""
    return (
        "CITATION FORMAT:\n"
        "- Cite every claim immediately after the sentence, not in a bibliography.\n"
        "- Format: [N] <Document Title> — Page <number>\n"
        f"- Example:\n{CITATION_EXAMPLE}\n"
        "- If the page number is unknown, write: [N] <Document Title> — (page unknown)\n"
    )


def get_fallback_instructions() -> str:
    """Instructions for when context is insufficient."""
    return (
        "FALLBACK RULE:\n"
        "If you cannot find a relevant answer in the context, respond with EXACTLY:\n"
        f'"{FALLBACK_RESPONSE}"\n'
        "Do NOT attempt to answer from general knowledge.\n"
    )


def build_rag_prompt(
    context_chunks: list[dict],
    user_question: str,
) -> str:
    """
    Assemble the full RAG prompt with context, question, and all rules.

    Args:
        context_chunks: List of retrieved chunk dicts with keys:
                        {text, source, page, title, chunk_id}
        user_question:  The user's (possibly translated) question string.

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    if not context_chunks:
        # Edge case: retrieval returned nothing.
        return (
            f"{get_system_prompt()}\n\n"
            "CONTEXT: [No relevant documents were retrieved.]\n\n"
            f"QUESTION: {user_question}\n\n"
            "ANSWER:"
        )

    # Format context blocks with numbered citations
    context_lines: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        title = chunk.get("title", "Unknown Document")
        source = chunk.get("source", "Unknown Source")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        context_lines.append(
            f"[{i}] {title} (Source: {source} | Page: {page})\n{text}"
        )

    context_block = "\n\n---\n\n".join(context_lines)

    return (
        f"{get_system_prompt()}\n\n"
        f"{get_citation_instructions()}\n\n"
        f"{get_fallback_instructions()}\n\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"QUESTION: {user_question}\n\n"
        "ANSWER:"
    )