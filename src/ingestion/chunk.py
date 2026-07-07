from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def build_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}_chunk_{index:04d}"


def chunk_document_record(
    record: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    chunk_counter = 1

    for page_record in record.get("pages", []):
        page = page_record.get("page")
        text = page_record.get("text", "").strip()
        if not text:
            continue

        split_texts = splitter.split_text(text)
        for piece in split_texts:
            piece = piece.strip()
            if not piece:
                continue

            chunks.append(
                {
                    "chunk_id": build_chunk_id(record["document_id"], chunk_counter),
                    "document_id": record["document_id"],
                    "page": page,
                    "title": record["title"],
                    "source": record["source_url"],
                    "language": record.get("language", "en"),
                    "text": piece,
                }
            )
            chunk_counter += 1

    return chunks


def build_chunks(cleaned_index_path: Path | None = None) -> list[dict]:
    settings = get_settings()
    cleaned_index_path = cleaned_index_path or (settings.data_cleaned_dir / "cleaned_index.json")

    records = json.loads(cleaned_index_path.read_text(encoding="utf-8"))
    all_chunks: list[dict] = []

    for record in records:
        chunks = chunk_document_record(
            record=record,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info("Chunked {} into {} chunks", record["title"], len(chunks))
        all_chunks.extend(chunks)

    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.data_processed_dir / "chunks.json"
    output_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved {} total chunks to {}", len(all_chunks), output_path)
    return all_chunks


if __name__ == "__main__":
    build_chunks()