from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from transformers import MarianMTModel, MarianTokenizer

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TranslationResult:
    source_language: str
    target_language: str
    original_text: str
    translated_text: str
    used_translation: bool
    model_name: str | None = None


def should_translate(source_language: str, target_language: str) -> bool:
    return source_language.strip().lower() != target_language.strip().lower()


def resolve_translation_model_name(source_language: str, target_language: str) -> str | None:
    settings = get_settings()
    configured = settings.translation_model_name

    if configured and configured != "auto":
        return configured

    # Safe default contract. Keep pluggable.
    mapping = {
        ("tl", "en"): "Helsinki-NLP/opus-mt-tc-big-tl-en",
        ("en", "tl"): "Helsinki-NLP/opus-mt-en-tl",
    }
    return mapping.get((source_language, target_language))


@lru_cache(maxsize=4)
def load_translation_components(model_name: str):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def marian_translate(text: str, model_name: str) -> str:
    tokenizer, model = load_translation_components(model_name)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**inputs, max_new_tokens=512)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return decoded[0].strip()


def translate_text(text: str, source_language: str, target_language: str) -> TranslationResult:
    if not text.strip():
        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            original_text=text,
            translated_text=text,
            used_translation=False,
            model_name=None,
        )

    if not should_translate(source_language, target_language):
        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            original_text=text,
            translated_text=text,
            used_translation=False,
            model_name=None,
        )

    model_name = resolve_translation_model_name(source_language, target_language)
    if not model_name:
        logger.warning(
            "No translation model configured for {} -> {}. Falling back to passthrough.",
            source_language,
            target_language,
        )
        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            original_text=text,
            translated_text=text,
            used_translation=False,
            model_name=None,
        )

    try:
        translated = marian_translate(text, model_name)
        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            original_text=text,
            translated_text=translated,
            used_translation=True,
            model_name=model_name,
        )
    except Exception as exc:
        logger.exception("Translation failed, falling back to passthrough: {}", exc)
        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            original_text=text,
            translated_text=text,
            used_translation=False,
            model_name=model_name,
        )


def translate_to_english(text: str, source_language: str) -> str:
    return translate_text(text, source_language, "en").translated_text


def translate_from_english(text: str, target_language: str) -> str:
    return translate_text(text, "en", target_language).translated_text