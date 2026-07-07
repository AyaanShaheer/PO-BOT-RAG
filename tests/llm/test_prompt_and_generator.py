from __future__ import annotations

from config.prompts import build_rag_prompt
from src.llm.generator import build_citation_block


def test_build_citation_block():
    chunks = [
        {
            "title": "Employment Ordinance",
            "page": 17,
            "source": "https://www.labour.gov.hk/eng/public/eo.htm",
        },
        {
            "title": "Foreign Domestic Helpers Guide",
            "page": 8,
            "source": "https://www.fdh.labour.gov.hk/en/home.html",
        },
    ]
    block = build_citation_block(chunks)
    assert "Employment Ordinance" in block
    assert "Page 17" in block


def test_build_rag_prompt_contains_question():
    prompt = build_rag_prompt(
        [{"title": "Doc", "source": "src", "page": 1, "text": "hello"}],
        "What is annual leave?"
    )
    assert "What is annual leave?" in prompt