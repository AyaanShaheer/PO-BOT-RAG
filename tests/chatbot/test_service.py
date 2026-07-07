from __future__ import annotations

from src.chatbot.service import format_final_answer


def test_format_final_answer_includes_sources():
    answer = format_final_answer(
        answer_text="Employees are entitled to annual leave [1].",
        confidence_score=0.84,
        confidence_label="High",
        sources=[
            {
                "title": "Employment Ordinance",
                "page": 17,
                "source": "https://www.labour.gov.hk/eng/public/eo.htm",
            }
        ],
        retrieval_latency="120.0 ms",
        generation_latency="1.42 s",
    )
    assert "Confidence" in answer
    assert "Employment Ordinance" in answer
    assert "1.42 s" in answer