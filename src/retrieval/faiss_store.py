from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from config.settings import get_settings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(
    index: faiss.Index,
    metadata: list[dict],
    index_dir: Path | None = None,
) -> None:
    settings = get_settings()
    index_dir = index_dir or settings.faiss_index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_faiss_index(index_dir: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    settings = get_settings()
    index_dir = index_dir or settings.faiss_index_dir

    index = faiss.read_index(str(index_dir / "index.faiss"))
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return index, metadata


def search_faiss_index(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if query_embeddings.ndim != 2:
        raise ValueError("Query embeddings must be 2D")
    return index.search(query_embeddings.astype("float32"), top_k)