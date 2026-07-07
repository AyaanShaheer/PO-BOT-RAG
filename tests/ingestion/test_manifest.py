from __future__ import annotations

from src.ingestion.collect import validate_manifest_entry


def test_validate_manifest_entry_accepts_official_labour_domain():
    entry = {
        "title": "Employment Ordinance",
        "source_url": "https://www.labour.gov.hk/eng/public/eo.htm",
        "source_type": "html",
        "language": "en",
    }
    validate_manifest_entry(entry)


def test_validate_manifest_entry_rejects_non_official_domain():
    entry = {
        "title": "Random mirror",
        "source_url": "https://randomsite.example/eo.htm",
        "source_type": "html",
        "language": "en",
    }
    try:
        validate_manifest_entry(entry)
    except ValueError as exc:
        assert "official" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for non-official domain")