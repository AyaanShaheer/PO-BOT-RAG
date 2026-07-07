from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0


def normalize_language_code(code: str) -> str:
    code = code.lower().strip()
    if code in {"tl", "fil"}:
        return "tl"
    if code == "en":
        return "en"
    return "en"


def detect_query_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "en"

    try:
        detected = detect(text)
        return normalize_language_code(detected)
    except LangDetectException:
        return "en"