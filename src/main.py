"""
main.py
-------
Application entry point.
Wires together configuration, logging, and the CLI chatbot.

Run modes:
  python src/main.py chat            → interactive chatbot (FR-17)
  python src/main.py ingest          → run ingestion + indexing pipeline (Phase 2-3)
  python src/main.py benchmark       → run evaluation suite (FR-19)
"""

from __future__ import annotations

import typer
from rich.console import Console

from config.settings import get_settings
from utils.logger import get_logger, setup_logging

app = typer.Typer(
    name="pobot",
    help="PoBot RAG — AI Assistant for Hong Kong Labour Regulations",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


def _bootstrap() -> None:
    """Initialise logging and directories before any module does real work."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level, app_env=settings.app_env)
    settings.ensure_directories()
    logger.info("PoBot starting | env={} | log_level={}", settings.app_env, settings.log_level)


@app.command()
def chat() -> None:
    """Launch the interactive CLI chatbot."""
    _bootstrap()
    from pathlib import Path
    settings = get_settings()

    required_files = [
        settings.faiss_index_dir / "index.faiss",
        settings.faiss_index_dir / "metadata.json",
        settings.bm25_index_dir / "bm25_corpus.json",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        console.print("[bold red]Missing retrieval artifacts.[/bold red]")
        for item in missing:
            console.print(f" - {item}")
        console.print("\nRun ingestion, preprocessing, chunking, and index building first.")
        raise typer.Exit(code=1)

    console.print("[bold teal]PoBot RAG[/bold teal] — Hong Kong Labour Regulations\n")
    from chatbot.cli import run_chat_loop  # noqa: PLC0415
    run_chat_loop()


@app.command()
def ingest() -> None:
    """Run document ingestion and indexing pipeline."""
    _bootstrap()
    console.print("[bold yellow]Running ingestion pipeline…[/bold yellow]")
    # Phase 2-3 entry point (stub until those phases are built)
    console.print("[dim]Ingestion pipeline not yet implemented (Phase 2).[/dim]")


@app.command()
def benchmark() -> None:
    """Run the evaluation benchmark suite."""
    _bootstrap()
    console.print("[bold blue]Running benchmark suite…[/bold blue]")

    from evaluation.benchmark import run_benchmark_suite  # noqa: PLC0415
    run_benchmark_suite()


if __name__ == "__main__":
    app()