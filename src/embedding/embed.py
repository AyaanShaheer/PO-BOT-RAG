from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from src.embedding.cache import (
    embeddings_cache_exists,
    load_cached_embeddings,
    save_cached_embeddings,
)
from utils.logger import get_logger
from utils.metrics import Timer

logger = get_logger(__name__)


def load_chunks(chunks_path: Path | None = None) -> list[dict]:
    settings = get_settings()
    chunks_path = chunks_path or (settings.data_processed_dir / "chunks.json")
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(model_name or settings.embedding_model_name)


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, 0), dtype="float32")

    model = model or get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def build_or_load_embeddings() -> tuple[np.ndarray, list[dict]]:
    settings = get_settings()
    cache_dir = settings.embeddings_dir

    if embeddings_cache_exists(cache_dir):
        logger.info("Loading cached embeddings from {}", cache_dir)
        embeddings, metadata = load_cached_embeddings(cache_dir)
        return embeddings.astype("float32"), metadata

    chunks = load_chunks()
    logger.info("No embedding cache found. Generating embeddings for {} chunks", len(chunks))

    with Timer("embedding") as t:
        model = get_embedding_model()
        embeddings = embed_chunks(chunks, model=model)

    save_cached_embeddings(cache_dir, embeddings, chunks)
    logger.info("Embeddings generated in {:.2f}s", t.elapsed_sec)
    return embeddings, chunks