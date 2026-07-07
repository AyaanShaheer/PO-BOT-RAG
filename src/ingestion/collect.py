from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import requests

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_SOURCE_TYPES = {"pdf", "html", "txt"}
OFFICIAL_HOST_SUFFIXES = (
    "labour.gov.hk",
    "fdh.labour.gov.hk",
    "eaa.labour.gov.hk",
    "elegislation.gov.hk",
)


@dataclass(slots=True)
class SourceDocument:
    title: str
    source_url: str
    source_type: Literal["pdf", "html", "txt"]
    language: str
    date: str | None = None
    document_id: str | None = None
    raw_path: str | None = None
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(
                f"Unsupported source_type={self.source_type!r}. "
                f"Allowed: {sorted(ALLOWED_SOURCE_TYPES)}"
            )
        if not self.document_id:
            self.document_id = build_document_id(self.source_url)


def build_document_id(source_url: str) -> str:
    digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "document"


def validate_manifest_entry(entry: dict) -> None:
    required = {"title", "source_url", "source_type", "language"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"Manifest entry missing required fields: {sorted(missing)}")

    url = entry["source_url"]
    if not any(host in url for host in OFFICIAL_HOST_SUFFIXES):
        raise ValueError(
            f"Source URL must point to an official Hong Kong government labour source: {url}"
        )

    if entry["source_type"] not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type: {entry['source_type']}")


def load_manifest(path: Path) -> list[SourceDocument]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    documents: list[SourceDocument] = []
    for entry in entries:
        validate_manifest_entry(entry)
        documents.append(SourceDocument(**entry))
    return documents


def guess_extension(source_type: str, url: str) -> str:
    if source_type == "pdf":
        return ".pdf"
    if source_type == "txt":
        return ".txt"
    if url.endswith(".html") or url.endswith(".htm"):
        return ".html"
    return ".html"


def save_raw_content(raw_dir: Path, filename: str, content: bytes) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename
    path.write_bytes(content)
    return path


def download_source(doc: SourceDocument, timeout: int = 30) -> bytes:
    headers = {
        "User-Agent": "PoBot-RAG/1.0 (+portfolio project; ingestion bot)"
    }
    response = requests.get(doc.source_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def collect_documents(manifest_path: Path | None = None) -> list[SourceDocument]:
    settings = get_settings()
    manifest_path = manifest_path or Path("data/source_manifest.json")
    documents = load_manifest(manifest_path)

    for doc in documents:
        logger.info("Collecting document: {}", doc.title)
        try:
            content = download_source(doc)
            ext = guess_extension(doc.source_type, doc.source_url)
            filename = sanitize_filename(f"{doc.document_id}_{doc.title}") + ext
            path = save_raw_content(settings.data_raw_dir, filename, content)

            doc.raw_path = str(path)
            doc.status = "collected"
            logger.info("Collected {} -> {}", doc.title, path)
        except Exception as exc:
            doc.status = f"failed: {exc}"
            logger.exception("Failed to collect {}", doc.title)

    return documents


def save_collection_metadata(documents: list[SourceDocument], output_path: Path | None = None) -> Path:
    output_path = output_path or Path("data/metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(doc) for doc in documents]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    docs = collect_documents()
    metadata_path = save_collection_metadata(docs)
    print(f"Saved metadata to {metadata_path}")