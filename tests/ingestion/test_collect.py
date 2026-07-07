from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.collect import (
    SourceDocument,
    build_document_id,
    sanitize_filename,
    save_raw_content,
)


class TestDocumentId:
    def test_build_document_id_is_stable(self):
        url = "https://www.labour.gov.hk/eng/public/eo.htm"
        doc_id_1 = build_document_id(url)
        doc_id_2 = build_document_id(url)
        assert doc_id_1 == doc_id_2

    def test_build_document_id_differs_for_different_urls(self):
        a = build_document_id("https://www.labour.gov.hk/eng/public/eo.htm")
        b = build_document_id("https://www.fdh.labour.gov.hk/en/home.html")
        assert a != b


class TestFilenameSanitization:
    def test_sanitize_filename_replaces_bad_chars(self):
        name = "Employment Ordinance: Guide / 2026?.html"
        assert sanitize_filename(name) == "employment_ordinance_guide_2026.html"

    def test_sanitize_filename_fallback(self):
        assert sanitize_filename("////") == "document"


class TestRawSave:
    def test_save_raw_content_html(self, tmp_path: Path):
        path = save_raw_content(
            raw_dir=tmp_path,
            filename="sample.html",
            content=b"<html><body>Hello</body></html>",
        )
        assert path.exists()
        assert path.read_bytes().startswith(b"<html>")

    def test_save_raw_content_creates_parent(self, tmp_path: Path):
        nested = tmp_path / "raw"
        path = save_raw_content(
            raw_dir=nested,
            filename="sample.txt",
            content=b"hello",
        )
        assert path.exists()


class TestSourceDocument:
    def test_source_document_defaults(self):
        doc = SourceDocument(
            title="Employment Ordinance",
            source_url="https://www.labour.gov.hk/eng/public/eo.htm",
            source_type="html",
            language="en",
        )
        assert doc.title == "Employment Ordinance"
        assert doc.language == "en"
        assert doc.date is None

    def test_source_document_rejects_invalid_source_type(self):
        with pytest.raises(ValueError):
            SourceDocument(
                title="Bad",
                source_url="https://example.com",
                source_type="docx",
                language="en",
            )