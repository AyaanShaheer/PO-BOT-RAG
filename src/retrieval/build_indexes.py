from __future__ import annotations

import json
from pathlib import Path

from config.settings import get_settings
from src.embedding.embed import build_or_load_embeddings
from src.retrieval.bm25_store import save_bm25_index
from src.retrieval.faiss_store import build_faiss_index, save_faiss_index
from utils.logger import get_logger
from utils.metrics import Timer

logger = get_logger(__name__)


def load_chunks(chunks_path: Path | None = None) -> list[dict]:
    settings = get_settings()
    chunks_path = chunks_path or (settings.data_processed_dir / "chunks.json")
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def build_all_indexes() -> None:
    settings = get_settings()
    chunks = load_chunks()

    with Timer("embedding_load_or_build") as t_embed:
        embeddings, metadata = build_or_load_embeddings()
    logger.info("Embedding stage completed in {:.2f}s", t_embed.elapsed_sec)

    with Timer("faiss_index_build") as t_faiss:
        faiss_index = build_faiss_index(embeddings)
        save_faiss_index(faiss_index, metadata, settings.faiss_index_dir)
    logger.info("FAISS stage completed in {:.2f}s", t_faiss.elapsed_sec)

    with Timer("bm25_index_build") as t_bm25:
        save_bm25_index(chunks, settings.bm25_index_dir)
    logger.info("BM25 stage completed in {:.2f}s", t_bm25.elapsed_sec)


if __name__ == "__main__":
    build_all_indexes()