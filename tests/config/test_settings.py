"""
tests/config/test_settings.py
------------------------------
Tests for Settings, config helpers, and prompt functions.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import os
import pytest

from config.settings import Settings, get_settings
from config.config import (
    get_groq_api_key,
    get_chunking_config,
    get_retrieval_config,
    is_embedding_cache_valid,
)
from config.prompts import (
    FALLBACK_RESPONSE,
    build_rag_prompt,
    get_citation_instructions,
    get_fallback_instructions,
    get_system_prompt,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure the lru_cache is cleared between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_settings(**overrides) -> Settings:
    """Helper: build Settings with controlled env, bypassing .env file."""
    defaults = dict(
        app_env="dev",
        log_level="DEBUG",
        groq_api_key=None,
        embedding_model_name="BAAI/bge-small-en-v1.5",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model_name="llama-3.1-8b-instant",
        chunk_size=500,
        chunk_overlap=80,
        hybrid_top_k=15,
        rerank_top_k=3,
        max_generation_latency_sec=3.0,
        max_retrieval_latency_ms=150.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ─── Settings: defaults ───────────────────────────────────────────────────────

class TestSettingsDefaults:

    def test_default_app_env_is_dev(self):
        s = make_settings()
        assert s.app_env == "dev"

    def test_default_models(self):
        s = make_settings()
        assert s.embedding_model_name == "BAAI/bge-small-en-v1.5"
        assert s.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert s.llm_model_name == "llama-3.1-8b-instant"

    def test_default_chunking(self):
        s = make_settings()
        assert s.chunk_size == 500
        assert s.chunk_overlap == 80

    def test_default_retrieval(self):
        s = make_settings()
        assert s.hybrid_top_k == 15
        assert s.rerank_top_k == 3

    def test_default_paths_are_pathlib(self):
        from pathlib import Path
        s = make_settings()
        assert isinstance(s.data_raw_dir, Path)
        assert isinstance(s.faiss_index_dir, Path)
        assert isinstance(s.metadata_path, Path)

    def test_is_prod_false_in_dev(self):
        s = make_settings(app_env="dev")
        assert not s.is_prod

    def test_is_test_true_in_test(self):
        s = make_settings(app_env="test")
        assert s.is_test


# ─── Settings: validation ─────────────────────────────────────────────────────

class TestSettingsValidation:

    def test_prod_requires_groq_key(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="GROQ_API_KEY"):
            make_settings(app_env="prod", groq_api_key=None)

    def test_prod_succeeds_with_groq_key(self):
        s = make_settings(app_env="prod", groq_api_key="sk-test-key")
        assert s.is_prod

    def test_dev_allows_missing_groq_key(self):
        s = make_settings(app_env="dev", groq_api_key=None)
        assert s.groq_api_key is None

    def test_chunk_overlap_must_be_lt_chunk_size(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="chunk_overlap"):
            make_settings(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_equal_to_chunk_size_fails(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_settings(chunk_size=200, chunk_overlap=200)

    def test_rerank_top_k_cannot_exceed_hybrid(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="rerank_top_k"):
            make_settings(hybrid_top_k=5, rerank_top_k=10)

    def test_chunk_size_must_be_positive(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_settings(chunk_size=0)

    def test_invalid_app_env_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_settings(app_env="staging")  # type: ignore[arg-type]

    def test_negative_generation_latency_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_settings(max_generation_latency_sec=-1.0)


# ─── Settings: env override ───────────────────────────────────────────────────

class TestSettingsEnvOverride:

    def test_embedding_model_override(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
        s = Settings()
        assert s.embedding_model_name == "BAAI/bge-large-en-v1.5"

    def test_chunk_size_override(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "1000")
        s = Settings()
        assert s.chunk_size == 1000

    def test_hybrid_top_k_override(self, monkeypatch):
        monkeypatch.setenv("HYBRID_TOP_K", "20")
        monkeypatch.setenv("RERANK_TOP_K", "5")
        s = Settings()
        assert s.hybrid_top_k == 20
        assert s.rerank_top_k == 5


# ─── Config helpers ───────────────────────────────────────────────────────────

class TestConfigHelpers:

    def test_get_groq_key_raises_when_missing(self):
        s = make_settings(app_env="dev", groq_api_key=None)
        with pytest.raises(RuntimeError, match="Groq API key is not configured"):
            get_groq_api_key(settings=s)

    def test_get_groq_key_returns_key(self):
        s = make_settings(groq_api_key="sk-test-abc")
        assert get_groq_api_key(settings=s) == "sk-test-abc"

    def test_chunking_config_returns_dict(self):
        s = make_settings(chunk_size=400, chunk_overlap=50)
        cfg = get_chunking_config(settings=s)
        assert cfg == {"chunk_size": 400, "chunk_overlap": 50}

    def test_retrieval_config_returns_dict(self):
        s = make_settings(hybrid_top_k=10, rerank_top_k=2)
        cfg = get_retrieval_config(settings=s)
        assert cfg == {"hybrid_top_k": 10, "rerank_top_k": 2}

    def test_embedding_cache_valid_false_when_no_file(self, tmp_path, monkeypatch):
        # Override the faiss dir to a known-empty tmp dir
        s = make_settings()
        monkeypatch.setattr(s, "faiss_index_dir", tmp_path / "faiss")
        assert not is_embedding_cache_valid(settings=s)

    def test_embedding_cache_valid_true_when_file_exists(self, tmp_path, monkeypatch):
        faiss_dir = tmp_path / "faiss"
        faiss_dir.mkdir()
        (faiss_dir / "index.faiss").write_bytes(b"")
        s = make_settings()
        monkeypatch.setattr(s, "faiss_index_dir", faiss_dir)
        assert is_embedding_cache_valid(settings=s)


# ─── Prompts ──────────────────────────────────────────────────────────────────

class TestPrompts:

    def test_system_prompt_contains_no_hallucinate_rule(self):
        prompt = get_system_prompt()
        assert "Do NOT fabricate" in prompt or "do not" in prompt.lower()

    def test_system_prompt_contains_fallback_text(self):
        prompt = get_system_prompt()
        assert FALLBACK_RESPONSE[:40] in prompt

    def test_citation_instructions_contain_format(self):
        ci = get_citation_instructions()
        assert "[N]" in ci
        assert "Page" in ci

    def test_fallback_instructions_contain_exact_fallback(self):
        fi = get_fallback_instructions()
        assert FALLBACK_RESPONSE in fi

    def test_build_rag_prompt_with_chunks(self):
        chunks = [
            {"title": "Employment Ordinance", "source": "cap57.pdf", "page": 17, "text": "Workers are entitled to 7 days annual leave after 12 months.", "chunk_id": "c1"},
        ]
        prompt = build_rag_prompt(chunks, "What are annual leave rules?")
        assert "Employment Ordinance" in prompt
        assert "annual leave" in prompt
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt

    def test_build_rag_prompt_empty_chunks_triggers_no_context(self):
        prompt = build_rag_prompt([], "What are annual leave rules?")
        assert "No relevant documents were retrieved" in prompt

    def test_build_rag_prompt_includes_all_citations(self):
        chunks = [
            {"title": "Doc A", "source": "a.pdf", "page": 1, "text": "Text A", "chunk_id": "a1"},
            {"title": "Doc B", "source": "b.pdf", "page": 5, "text": "Text B", "chunk_id": "b1"},
        ]
        prompt = build_rag_prompt(chunks, "Test question")
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Doc A" in prompt
        assert "Doc B" in prompt