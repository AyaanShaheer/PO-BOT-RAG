from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embedding.cache import (
    embeddings_cache_exists,
    load_cached_embeddings,
    save_cached_embeddings,
)


def test_save_and_load_cached_embeddings(tmp_path: Path):
    embeddings = np.random.rand(3, 4).astype("float32")
    metadata = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]

    save_cached_embeddings(tmp_path, embeddings, metadata)

    assert embeddings_cache_exists(tmp_path)

    loaded_embeddings, loaded_metadata = load_cached_embeddings(tmp_path)
    assert loaded_embeddings.shape == (3, 4)
    assert loaded_metadata[0]["chunk_id"] == "c1"


def test_embeddings_cache_exists_false_when_missing(tmp_path: Path):
    assert not embeddings_cache_exists(tmp_path)