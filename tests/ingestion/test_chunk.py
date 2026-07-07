from __future__ import annotations

from src.ingestion.chunk import (
    build_chunk_id,
    chunk_document_record,
)


def test_build_chunk_id_is_stable():
    chunk_id_1 = build_chunk_id("doc_abc", 1)
    chunk_id_2 = build_chunk_id("doc_abc", 1)
    assert chunk_id_1 == chunk_id_2


def test_chunk_document_record_preserves_metadata():
    record = {
        "document_id": "doc_123",
        "title": "Employment Ordinance",
        "source_url": "https://www.labour.gov.hk/eng/public/eo.htm",
        "language": "en",
        "pages": [
            {"page": 1, "text": "This is page one. " * 80},
        ],
    }

    chunks = chunk_document_record(record, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 1
    assert chunks[0]["document_id"] == "doc_123"
    assert chunks[0]["title"] == "Employment Ordinance"
    assert chunks[0]["source"] == "https://www.labour.gov.hk/eng/public/eo.htm"
    assert "chunk_id" in chunks[0]


def test_chunk_document_record_handles_empty_pages():
    record = {
        "document_id": "doc_empty",
        "title": "Empty Doc",
        "source_url": "https://www.labour.gov.hk/eng/public/eo.htm",
        "language": "en",
        "pages": [],
    }
    chunks = chunk_document_record(record, chunk_size=300, chunk_overlap=50)
    assert chunks == []