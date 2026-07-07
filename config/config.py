"""
config.py
---------
Higher-level helpers that compose settings into ready-to-use structures.
Keeps all 'what does dev/prod mean behaviourally' logic in one place so
the rest of the codebase just imports helpers, not raw settings fields.
"""

from __future__ import annotations

from config.settings import Settings, get_settings


def get_log_level(settings: Settings | None = None) -> str:
    """Return effective log level string for Loguru."""
    s = settings or get_settings()
    return s.log_level


def get_groq_api_key(settings: Settings | None = None) -> str:
    """
    Return the Groq API key.
    Raises RuntimeError with a helpful message if not configured.
    This is the single place that enforces the 'must have key to call LLM' rule.
    """
    s = settings or get_settings()
    if not s.groq_api_key:
        raise RuntimeError(
            "Groq API key is not configured. "
            "Set GROQ_API_KEY in your .env file or environment. "
            f"Current APP_ENV={s.app_env!r}."
        )
    return s.groq_api_key


def is_embedding_cache_valid(settings: Settings | None = None) -> bool:
    """
    Return True if a cached embedding file already exists on disk.
    Used by embed.py to skip regeneration (FR-6).
    """
    s = settings or get_settings()
    # We check for the FAISS index file as the canonical cache marker.
    faiss_marker = s.faiss_index_dir / "index.faiss"
    return faiss_marker.exists()


def get_chunking_config(settings: Settings | None = None) -> dict[str, int]:
    """Return chunking parameters as a plain dict for the text splitter."""
    s = settings or get_settings()
    return {
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
    }


def get_retrieval_config(settings: Settings | None = None) -> dict[str, int]:
    """Return retrieval parameters as a plain dict."""
    s = settings or get_settings()
    return {
        "hybrid_top_k": s.hybrid_top_k,
        "rerank_top_k": s.rerank_top_k,
    }