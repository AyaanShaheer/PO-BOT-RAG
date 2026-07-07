from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import get_settings


def tokenize_for_bm25(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def build_bm25_index(corpus: list[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize_for_bm25(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def save_bm25_index(
    chunks: list[dict],
    index_dir: Path | None = None,
) -> None:
    settings = get_settings()
    index_dir = index_dir or settings.bm25_index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
    payload = {
        "tokenized_corpus": tokenized_corpus,
        "metadata": chunks,
    }
    (index_dir / "bm25_corpus.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_bm25_index(index_dir: Path | None = None) -> tuple[BM25Okapi, list[dict]]:
    settings = get_settings()
    index_dir = index_dir or settings.bm25_index_dir

    payload = json.loads((index_dir / "bm25_corpus.json").read_text(encoding="utf-8"))
    bm25 = BM25Okapi(payload["tokenized_corpus"])
    return bm25, payload["metadata"]


def search_bm25_index(
    bm25: BM25Okapi,
    query: str,
    top_k: int,
) -> list[tuple[int, float]]:
    tokenized_query = tokenize_for_bm25(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked_indices]