"""
settings.py
-----------
Single source of truth for all runtime configuration.
Uses pydantic-settings for typed, validated, environment-driven config.

Design decisions:
  - GROQ_API_KEY is Optional in dev/test but enforced at validation time in prod.
  - All paths are returned as pathlib.Path so callers never do string juggling.
  - Numeric bounds are validated with @field_validator so bad values fail fast.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable, validated application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # silently drop unknown env vars
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── External services ────────────────────────────────────────────────────
    groq_api_key: Optional[str] = Field(default=None, repr=False)

    # ── Models ───────────────────────────────────────────────────────────────
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model_name: str = "llama-3.1-8b-instant"
    translation_model_name: str = "Helsinki-NLP/opus-mt-en-tl"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)

    # ── Retrieval ────────────────────────────────────────────────────────────
    hybrid_top_k: int = Field(default=15, gt=0)
    rerank_top_k: int = Field(default=3, gt=0)

    # ── Performance ──────────────────────────────────────────────────────────
    max_generation_latency_sec: float = Field(default=3.0, gt=0)
    max_retrieval_latency_ms: float = Field(default=150.0, gt=0)

    # ── Paths (relative to project root, resolvable at runtime) ──────────────
    data_raw_dir: Path = Path("data/raw")
    data_cleaned_dir: Path = Path("data/cleaned")
    data_processed_dir: Path = Path("data/processed")
    embeddings_dir: Path = Path("embeddings")
    faiss_index_dir: Path = Path("indexes/faiss")
    bm25_index_dir: Path = Path("indexes/bm25")
    metadata_path: Path = Path("data/metadata.json")

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v: int, info) -> int:
        # info.data is populated with already-validated fields (chunk_size)
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be strictly less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_must_be_lte_hybrid(cls, v: int, info) -> int:
        hybrid_top_k = info.data.get("hybrid_top_k", 15)
        if v > hybrid_top_k:
            raise ValueError(
                f"rerank_top_k ({v}) cannot exceed hybrid_top_k ({hybrid_top_k})"
            )
        return v

    @model_validator(mode="after")
    def require_groq_key_in_prod(self) -> "Settings":
        if self.is_prod and not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be set when APP_ENV=prod. "
                "Export it or add it to your .env file."
            )
        return self

    def ensure_directories(self) -> None:
        """
        Create all required data directories if they do not yet exist.
        Called explicitly by main.py on startup — NOT in __init__ (keeps Settings pure).
        """
        dirs = [
            self.data_raw_dir,
            self.data_cleaned_dir,
            self.data_processed_dir,
            self.embeddings_dir,
            self.faiss_index_dir,
            self.bm25_index_dir,
            self.metadata_path.parent,   # data/ for metadata.json
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a module-level cached Settings singleton.
    Use this everywhere: `from config.settings import get_settings`

    The cache means .env is read exactly once per process.
    In tests, call get_settings.cache_clear() then monkeypatch env vars.
    """
    return Settings()