from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config.prompts import FALLBACK_RESPONSE
from config.settings import get_settings
from src.chatbot.service import run_chat_request
from src.evaluation.io import load_jsonl, write_jsonl_record
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass(slots=True)
class BenchmarkCase:
    question: str
    expected_documents: list[str]
    expected_keywords: list[str]
    language: str = "en"


@dataclass(slots=True)
class BenchmarkResult:
    question: str
    expected_documents: list[str]
    retrieved_documents: list[str]
    retrieval_hit: bool
    keyword_hit: bool
    fallback_used: bool
    confidence_score: float
    confidence_label: str
    retrieval_latency_ms: float
    generation_latency_sec: float
    passed: bool
    answer_preview: str


def load_benchmark_cases(path: Path) -> list[BenchmarkCase]:
    rows = load_jsonl(path)
    return [BenchmarkCase(**row) for row in rows]


def check_expected_documents_hit(
    retrieved_chunks: list[dict],
    expected_documents: list[str],
) -> bool:
    retrieved_titles = {
        chunk.get("title", "").strip().lower()
        for chunk in retrieved_chunks
    }
    expected_titles = {
        item.strip().lower() for item in expected_documents
    }
    return any(title in retrieved_titles for title in expected_titles)


def check_expected_keywords_hit(answer_text: str, expected_keywords: list[str]) -> bool:
    lowered = answer_text.lower()
    return all(keyword.lower() in lowered for keyword in expected_keywords)


def evaluate_case_result(
    case: BenchmarkCase,
    retrieved_chunks: list[dict],
    answer_text: str,
    confidence_score: float,
    retrieval_latency_ms: float,
    generation_latency_sec: float,
) -> BenchmarkResult:
    retrieval_hit = check_expected_documents_hit(
        retrieved_chunks=retrieved_chunks,
        expected_documents=case.expected_documents,
    )
    keyword_hit = check_expected_keywords_hit(
        answer_text=answer_text,
        expected_keywords=case.expected_keywords,
    )
    fallback_used = FALLBACK_RESPONSE.lower() in answer_text.lower()

    confidence_label = (
        "High" if confidence_score >= 0.75
        else "Medium" if confidence_score >= 0.45
        else "Low"
    )

    passed = (
        retrieval_hit
        and keyword_hit
        and not fallback_used
        and retrieval_latency_ms <= 150.0
        and generation_latency_sec <= 3.0
    )

    retrieved_documents = []
    seen = set()
    for chunk in retrieved_chunks:
        title = chunk.get("title", "Unknown")
        if title not in seen:
            seen.add(title)
            retrieved_documents.append(title)

    return BenchmarkResult(
        question=case.question,
        expected_documents=case.expected_documents,
        retrieved_documents=retrieved_documents,
        retrieval_hit=retrieval_hit,
        keyword_hit=keyword_hit,
        fallback_used=fallback_used,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_sec=generation_latency_sec,
        passed=passed,
        answer_preview=answer_text[:220],
    )


def render_benchmark_table(results: list[BenchmarkResult]) -> Table:
    table = Table(title="Benchmark Results")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Question", style="green")
    table.add_column("Retrieval", style="yellow")
    table.add_column("Keywords", style="yellow")
    table.add_column("Confidence", style="magenta")
    table.add_column("Latency", style="blue")
    table.add_column("Pass", style="bold")

    for idx, result in enumerate(results, start=1):
        latency = f"{result.retrieval_latency_ms:.1f} ms / {result.generation_latency_sec:.2f} s"
        table.add_row(
            str(idx),
            result.question[:42] + ("..." if len(result.question) > 42 else ""),
            "✅" if result.retrieval_hit else "❌",
            "✅" if result.keyword_hit else "❌",
            f"{result.confidence_score:.2f} ({result.confidence_label})",
            latency,
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
        )
    return table


def run_benchmark_suite(
    cases_path: Path | None = None,
    output_path: Path | None = None,
) -> list[BenchmarkResult]:
    settings = get_settings()
    cases_path = cases_path or Path("data/benchmarks/benchmark_cases.jsonl")
    output_path = output_path or Path("data/benchmarks/benchmark_results.jsonl")

    cases = load_benchmark_cases(cases_path)
    results: list[BenchmarkResult] = []

    logger.info("Loaded {} benchmark cases", len(cases))

    for idx, case in enumerate(cases, start=1):
        logger.info("Running benchmark case {}: {}", idx, case.question)

        chat_response = run_chat_request(case.question)
        retrieval_latency_str = chat_response.query_result.latency.get("retrieval", "0 ms")
        generation_latency_str = chat_response.generation_latency

        retrieval_latency_ms = float(retrieval_latency_str.replace(" ms", "").replace(" s", ""))
        generation_latency_sec = float(generation_latency_str.replace(" s", "").replace(" ms", ""))

        result = evaluate_case_result(
            case=case,
            retrieved_chunks=chat_response.query_result.retrieved_chunks,
            answer_text=chat_response.answer_text,
            confidence_score=chat_response.query_result.confidence_score,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_sec=generation_latency_sec,
        )
        results.append(result)
        write_jsonl_record(output_path, asdict(result))

    console.print()
    console.print(render_benchmark_table(results))

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    logger.info("Benchmark complete: {}/{} passed", passed, total)

    return results