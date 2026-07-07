from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
from bs4 import BeautifulSoup

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

BOILERPLATE_PATTERNS = [
    r"Skip to content",
    r"Home\s*\|",
    r"Contact us",
    r"What's New",
    r"WELCOME MESSAGE",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def remove_common_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def extract_text_from_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    target = main if main else soup.body or soup
    text = target.get_text(separator="\n", strip=True)
    return normalize_whitespace(text)


def extract_text_from_txt(content: bytes) -> str:
    return normalize_whitespace(content.decode("utf-8", errors="ignore"))


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[dict]]:
    doc = fitz.open(pdf_path)
    pages: list[dict] = []
    chunks: list[str] = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = normalize_whitespace(text)
        if text:
            pages.append({"page": i, "text": text})
            chunks.append(f"[Page {i}]\n{text}")

    return "\n\n".join(chunks).strip(), pages


def clean_text(raw_text: str) -> str:
    return remove_common_boilerplate(normalize_whitespace(raw_text))


def preprocess_file(raw_path: Path) -> dict:
    suffix = raw_path.suffix.lower()

    if suffix in {".html", ".htm"}:
        raw_bytes = raw_path.read_bytes()
        extracted = extract_text_from_html(raw_bytes)
        pages = [{"page": None, "text": extracted}]
    elif suffix == ".txt":
        raw_bytes = raw_path.read_bytes()
        extracted = extract_text_from_txt(raw_bytes)
        pages = [{"page": None, "text": extracted}]
    elif suffix == ".pdf":
        extracted, pages = extract_text_from_pdf(raw_path)
    else:
        raise ValueError(f"Unsupported file type: {raw_path.suffix}")

    cleaned = clean_text(extracted)
    return {
        "raw_path": str(raw_path),
        "cleaned_text": cleaned,
        "pages": pages,
    }


def preprocess_all(metadata_path: Path | None = None) -> list[dict]:
    settings = get_settings()
    metadata_path = metadata_path or settings.metadata_path

    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs: list[dict] = []

    settings.data_cleaned_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        raw_path = record.get("raw_path")
        if not raw_path or not Path(raw_path).exists():
            logger.warning("Skipping document with missing raw_path: {}", record.get("title"))
            continue

        logger.info("Preprocessing {}", record["title"])
        processed = preprocess_file(Path(raw_path))

        output = {
            "document_id": record["document_id"],
            "title": record["title"],
            "source_url": record["source_url"],
            "source_type": record["source_type"],
            "language": record["language"],
            "date": record.get("date"),
            "raw_path": raw_path,
            "cleaned_text_path": str(
                settings.data_cleaned_dir / f"{record['document_id']}.txt"
            ),
            "structured_path": str(
                settings.data_cleaned_dir / f"{record['document_id']}.json"
            ),
            "pages": processed["pages"],
        }

        Path(output["cleaned_text_path"]).write_text(
            processed["cleaned_text"],
            encoding="utf-8",
        )
        Path(output["structured_path"]).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs.append(output)

    cleaned_index_path = settings.data_cleaned_dir / "cleaned_index.json"
    cleaned_index_path.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return outputs


if __name__ == "__main__":
    preprocess_all()