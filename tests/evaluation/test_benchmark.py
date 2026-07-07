from __future__ import annotations

from src.evaluation.benchmark import (
    BenchmarkCase,
    check_expected_documents_hit,
    evaluate_case_result,
)


def test_check_expected_documents_hit_true():
    retrieved = [
        {"title": "Employment Ordinance", "source": "https://www.labour.gov.hk/eng/public/eo.htm"},
        {"title": "Foreign Domestic Helpers Guide", "source": "https://www.fdh.labour.gov.hk/en/home.html"},
    ]
    expected = ["Employment Ordinance"]
    assert check_expected_documents_hit(retrieved, expected) is True


def test_check_expected_documents_hit_false():
    retrieved = [
        {"title": "Employment Agencies Portal", "source": "https://www.eaa.labour.gov.hk/en/home.html"},
    ]
    expected = ["Employment Ordinance"]
    assert check_expected_documents_hit(retrieved, expected) is False


def test_evaluate_case_result():
    case = BenchmarkCase(
        question="What are annual leave rules?",
        expected_documents=["Employment Ordinance"],
        expected_keywords=["annual leave"],
        language="en",
    )
    result = evaluate_case_result(
        case=case,
        retrieved_chunks=[
            {"title": "Employment Ordinance", "text": "Employees are entitled to annual leave."}
        ],
        answer_text="Employees are entitled to annual leave [1].",
        confidence_score=0.81,
        retrieval_latency_ms=120.0,
        generation_latency_sec=1.8,
    )
    assert result.retrieval_hit is True
    assert result.passed is True