from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def embeddings_cache_exists(cache_dir: Path) -> bool:
    return (
        (cache_dir / "embeddings.npy").exists()
        and (cache_dir / "metadata.json").exists()
    )


def save_cached_embeddings(
    cache_dir: Path,
    embeddings: np.ndarray,
    metadata: list[dict],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings)
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_cached_embeddings(cache_dir: Path) -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(cache_dir / "embeddings.npy")
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    return embeddings, metadata