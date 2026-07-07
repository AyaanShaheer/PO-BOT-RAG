from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from config.prompts import FALLBACK_RESPONSE
from src.chatbot.service import run_chat_request, stream_answer_tokens
from src.llm.postprocess import postprocess_answer
from src.retrieval.pipeline import run_query_pipeline
from src.translation.translate import translate_from_english
from utils.logger import get_logger
from utils.metrics import Timer

console = Console()
logger = get_logger(__name__)


def render_sources_table(chunks: list[dict]) -> Table:
    table = Table(title="Sources")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Document", style="green")
    table.add_column("Page", style="yellow")
    table.add_column("Source", style="magenta")

    seen: set[tuple[str, str]] = set()
    row_num = 1

    for chunk in chunks:
        title = chunk.get("title", "Unknown")
        page = str(chunk.get("page", "unknown"))
        source = chunk.get("source", "Unknown")
        key = (title, page)
        if key in seen:
            continue
        seen.add(key)

        table.add_row(str(row_num), title, page, source)
        row_num += 1

    return table


def stream_to_console(question: str, retrieved_chunks: list[dict], detected_language: str) -> tuple[str, float]:
    collected = ""

    with Timer("generation") as t_generation:
        with Live(Panel("", title="Answer", border_style="blue"), console=console, refresh_per_second=12) as live:
            for token in stream_answer_tokens(question, retrieved_chunks):
                collected += token
                live.update(Panel(collected, title="Answer", border_style="blue"), refresh=True)

    answer = postprocess_answer(collected)
    if detected_language != "en":
        answer = translate_from_english(answer, detected_language)

    return answer, t_generation.elapsed_sec


def run_chat_loop() -> None:
    console.print(Panel.fit("PoBot RAG — Ask about Hong Kong Labour Regulations", style="bold cyan"))
    console.print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = console.input("[bold green]Ask > [/bold green]").strip()

        if question.lower() in {"exit", "quit"}:
            console.print("\n[bold yellow]Goodbye.[/bold yellow]")
            break

        if not question:
            console.print("[red]Please enter a question.[/red]")
            continue

        try:
            pipeline_result = run_query_pipeline(question)

            if pipeline_result.fallback_used:
                answer = FALLBACK_RESPONSE
                generation_sec = 0.0
                console.print("\n[bold blue]Answer[/bold blue]")
                console.print(Panel(answer, expand=False))
            else:
                console.print()
                answer, generation_sec = stream_to_console(
                    question=pipeline_result.normalized_query,
                    retrieved_chunks=pipeline_result.retrieved_chunks,
                    detected_language=pipeline_result.detected_language,
                )

            console.print(
                f"\n[bold]Confidence:[/bold] "
                f"{pipeline_result.confidence_score:.2f} ({pipeline_result.confidence_label})"
            )
            console.print(f"[bold]Detected Language:[/bold] {pipeline_result.detected_language}")
            console.print(f"[bold]Retrieval Latency:[/bold] {pipeline_result.latency.get('retrieval', 'N/A')}")
            console.print(f"[bold]Generation Latency:[/bold] {generation_sec:.2f} s")

            if pipeline_result.retrieved_chunks:
                console.print()
                console.print(render_sources_table(pipeline_result.retrieved_chunks[:3]))

        except Exception as exc:
            logger.exception("CLI query failed")
            console.print(f"[red]Error:[/red] {exc}")

        console.print()